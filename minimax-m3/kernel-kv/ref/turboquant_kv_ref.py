"""turboquant_kv_ref.py -- the numerically-authoritative CPU reference (executable
spec + parity oracle) for the MiniMax-M3 sub-fp8 KV-cache TurboQuant OpBackend kernel.

This module is PURE PyTorch, runs on CPU (NO habana_frameworks, NO HPU, NO card), and
defines the EXACT math the two compiled OpBackend ops emit:

  * kv_rotate_quant  (the KV-WRITE op, hpu_kvq_ext::kv_rotate_quant):
        rotate K,V by a per-head random rotation (Walsh-Hadamard / structured butterfly,
        calibration-free) to SPREAD per-coordinate energy, then SCALAR low-bit quantize
        (K -> fp8 e4m3 ; V -> 4-bit or 2-bit signed int, per-(token,head) symmetric scale).
        Returns packed codes + scales.

  * kv_dequant_unrotate  (the KV-READ op, folded into the SDPA path):
        dequant the codes back to bf16, then INVERSE-rotate (rotation is orthogonal so
        inverse == transpose == apply the SAME Hadamard again / 2x for the H2 normalised
        case -> cheap), yielding K,V ready for the (fp8 or bf16) SDPA.

TurboQuant (Google, arXiv 2504.19874): a data-OBLIVIOUS random rotation makes every
coordinate marginal ~Gaussian, so a single per-vector scalar scale quantizes near the
information-theoretic optimum with NO calibration. The vLLM impl substitutes the cheap
structured Walsh-Hadamard transform (WHT) for the paper's dense Haar rotation; we do the
same because the WHT is a butterfly == a sequence of ±1 GEMMs == maps to the bridge
`batch_gemm` GUID (see kv_rotate_quant_backend_shared.h). The WHT is its OWN inverse
(H @ H = n*I), so the read-side "inverse rotation" is the identical transform divided by n
-- we fold the 1/sqrt(n) normalisation into BOTH directions so each is an exact orthogonal
involution (H_n = WHT/sqrt(n), H_n @ H_n = I).

Primary config **k8v4** (K fp8, V int4) is the upstream-blessed safe knee (review
quant-kv-turboquant.md). A **2bit** "lightest" mode (K fp8, V int2) is ALSO provided as
an option for the 1M-fit case -- the review rejects 2-bit *values* on quality grounds, so
it is built + measured but NOT a default.

Everything is shape-driven (no model constants). M3 GQA dims are H_kv=4, head_dim=128.

================================  PARITY CONTRACT  ============================
The compiled op's AddNode node-chain reproduces these functions step-for-step on the
device; the CPU test (tests/test_turboquant_kv_cpu.py) diffs THIS reference's
reconstruction + attention against fp8-KV on synthetic M3-shaped tensors. The on-card
test (tests/test_turboquant_kv_op.py, GATED) diffs the device op vs THIS reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

# fp8 e4m3 finite range (the K quant target; matches convert_to_fp8 e4m3fn on Gaudi).
FP8_E4M3_MAX = 448.0


# =============================================================================
# Walsh-Hadamard transform (the calibration-free random rotation).
# =============================================================================
def hadamard_matrix(n: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Normalised Hadamard matrix H_n (n a power of two) with H_n @ H_n == I.

    Built by the Sylvester recursion then scaled by 1/sqrt(n). Because H_n is symmetric
    and orthogonal, it is its OWN inverse -- the read side applies the SAME H_n. This is
    the structured stand-in for TurboQuant's random rotation; on M3 head_dim=128 = 2^7 it
    is exact (no padding needed). For a non-pow2 head_dim the kernel pads to the next 2^k
    (documented; M3 never hits it)."""
    assert n > 0 and (n & (n - 1)) == 0, f"Hadamard needs a power-of-two size, got {n}"
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    H = H / math.sqrt(n)  # normalise so H @ H == I (orthogonal involution)
    return H.to(dtype=dtype, device=device)


def _random_signs(n: int, seed: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Optional per-coordinate random sign flip diag(s), s in {-1,+1}, folded into the
    rotation as H @ diag(s). diag(s) is its own inverse, so (H@D) inverse = D@H = (H@D)^T.
    Calibration-free (seed is a fixed constant, NOT data-derived). Spreading is already
    achieved by H alone for our marginals; the sign diag is a cheap extra de-correlation
    the paper allows. Default OFF (signs=all-ones) so the transform is exactly H_n and the
    kernel's pure batch_gemm matches; enable by passing use_signs=True symmetrically on
    both sides."""
    g = torch.Generator().manual_seed(seed)
    s = torch.randint(0, 2, (n,), generator=g, dtype=torch.int32).to(dtype) * 2 - 1
    return s.to(device)


def make_rotation(head_dim: int, use_signs: bool = False, seed: int = 0x5151,
                  dtype=torch.float32, device="cpu") -> torch.Tensor:
    """The per-head rotation matrix R (head_dim x head_dim), R @ R^T == I. R is shared by
    K and V and by all heads (data-oblivious). Returns R such that rotate(x) = x @ R and
    unrotate(y) = y @ R^T. With use_signs=False, R = H_n (symmetric -> R^T == R)."""
    H = hadamard_matrix(head_dim, dtype=dtype, device=device)
    if not use_signs:
        return H
    s = _random_signs(head_dim, seed, dtype=dtype, device=device)
    return H * s.view(1, -1)  # H @ diag(s); columns scaled


# =============================================================================
# Scalar symmetric low-bit quant helpers (per-(token,head) scale over head_dim).
# =============================================================================
def _absmax_scale(x: torch.Tensor, qmax: float, eps: float = 1e-12) -> torch.Tensor:
    """Per-row (last-dim) absmax scale so x/scale lands in [-qmax, qmax]. x is [..., D];
    scale is [..., 1]. This is exactly reduce_max(abs(x), dim=-1)/qmax in the kernel."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    return amax / qmax


def quant_fp8_e4m3(x_rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """K path: bf16 rotated K -> fp8 e4m3 + per-(token,head) scale. Mirrors
    convert_to_fp8(e4m3) with an explicit reciprocal scale (the bridge convert_to_fp8
    consumes 1/scale). Returns (k_fp8 [.,D] float8_e4m3fn, scale [.,1] float32)."""
    scale = _absmax_scale(x_rot.float(), FP8_E4M3_MAX)           # [.,1]
    q = (x_rot.float() / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    k_fp8 = q.to(torch.float8_e4m3fn)
    return k_fp8, scale.float()


def dequant_fp8_e4m3(k_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """K read: fp8 -> bf16-domain float (still rotated). out = fp8.float() * scale."""
    return k_fp8.float() * scale.float()


def quant_int_sym(x_rot: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """V path: bf16 rotated V -> signed b-bit int code + per-(token,head) scale.
    Symmetric, round-half-to-even, clamp to [-(2^(b-1)-1), 2^(b-1)-1] (symmetric range;
    the -2^(b-1) code is unused so the dequant is exactly symmetric -- matches the kernel's
    clamp bounds). Returns (codes int8 in [-qmax,qmax], scale [.,1] float32)."""
    assert bits in (4, 2), f"V quant supports 4 or 2 bits, got {bits}"
    qmax = float((1 << (bits - 1)) - 1)                          # 4b->7, 2b->1
    scale = _absmax_scale(x_rot.float(), qmax)                   # [.,1]
    q = torch.round(x_rot.float() / scale).clamp(-qmax, qmax)
    return q.to(torch.int8), scale.float()


def dequant_int_sym(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """V read: int code -> bf16-domain float (still rotated). out = code.float()*scale."""
    return codes.float() * scale.float()


# =============================================================================
# Sub-byte PACK/UNPACK (storage layout; the HBM-footprint win lives here).
# =============================================================================
def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack the LAST dim of signed-int4 codes [...,D] (D%8==0) into int32 lanes
    [...,D/8], 8 nibbles/lane, little-endian nibble order. Two's-complement 4-bit so the
    device `reinterpret_cast`->`dequantize_4_bit` path round-trips. CPU mirror of the
    kernel's pack node."""
    *lead, D = codes.shape
    assert D % 8 == 0, f"int4 pack needs last dim %8==0, got {D}"
    c = (codes.to(torch.int32) & 0xF)                            # 4-bit two's complement
    c = c.reshape(*lead, D // 8, 8)
    shifts = (torch.arange(8, device=codes.device) * 4).view(*([1] * len(lead)), 1, 8)
    return (c << shifts).sum(dim=-1).to(torch.int32)             # [...,D/8] int32


def unpack_int4(packed: torch.Tensor, D: int) -> torch.Tensor:
    """Inverse of pack_int4: int32 lanes [...,D/8] -> signed-int4 codes [...,D] int8."""
    *lead, L = packed.shape
    assert L * 8 == D
    shifts = (torch.arange(8, device=packed.device) * 4).view(*([1] * len(lead)), 1, 8)
    nib = (packed.unsqueeze(-1) >> shifts) & 0xF                 # [...,D/8,8]
    nib = torch.where(nib >= 8, nib - 16, nib)                   # sign-extend 4->int
    return nib.reshape(*lead, D).to(torch.int8)


def pack_int2(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed-int2 codes (values in {-1,0,1}) [...,D] (D%16==0) into int32 lanes
    [...,D/16], 16 crumbs/lane. Two's-complement 2-bit."""
    *lead, D = codes.shape
    assert D % 16 == 0, f"int2 pack needs last dim %16==0, got {D}"
    c = (codes.to(torch.int32) & 0x3)
    c = c.reshape(*lead, D // 16, 16)
    shifts = (torch.arange(16, device=codes.device) * 2).view(*([1] * len(lead)), 1, 16)
    return (c << shifts).sum(dim=-1).to(torch.int32)


def unpack_int2(packed: torch.Tensor, D: int) -> torch.Tensor:
    *lead, L = packed.shape
    assert L * 16 == D
    shifts = (torch.arange(16, device=packed.device) * 2).view(*([1] * len(lead)), 1, 16)
    cr = (packed.unsqueeze(-1) >> shifts) & 0x3
    cr = torch.where(cr >= 2, cr - 4, cr)                        # sign-extend 2->int
    return cr.reshape(*lead, D).to(torch.int8)


# =============================================================================
# Top-level WRITE / READ (the two-op contract, exactly as the kernel composes them).
# =============================================================================
@dataclass
class KVQConfig:
    mode: str = "k8v4"          # "k8v4" (K fp8 / V int4) or "k8v2" (K fp8 / V int2)
    use_signs: bool = False     # add a random sign diag to the rotation (both sides)
    seed: int = 0x5151

    @property
    def v_bits(self) -> int:
        return 4 if self.mode == "k8v4" else 2


def kv_rotate_quant(k: torch.Tensor, v: torch.Tensor, cfg: KVQConfig):
    """KV-WRITE. k,v are [..., D] (D == head_dim) bf16/float (any leading [T,H] etc.).
    Returns a dict of stored artefacts (rotation is implicit/shared, not stored):
        k_fp8  : [...,D]      float8_e4m3fn   (rotated K, fp8)
        k_scale: [...,1]      float32
        v_packed: [...,D/8] (int4) or [...,D/16] (int2)  int32   (rotated V, packed)
        v_scale: [...,1]      float32
    The rotation R is regenerated deterministically from (D,cfg) on both sides; we never
    persist it (it's data-oblivious + tiny)."""
    D = k.shape[-1]
    R = make_rotation(D, use_signs=cfg.use_signs, seed=cfg.seed, device=k.device)
    k_rot = k.float() @ R
    v_rot = v.float() @ R
    k_fp8, k_scale = quant_fp8_e4m3(k_rot)
    v_codes, v_scale = quant_int_sym(v_rot, cfg.v_bits)
    if cfg.v_bits == 4:
        v_packed = pack_int4(v_codes)
    else:
        v_packed = pack_int2(v_codes)
    return {
        "k_fp8": k_fp8, "k_scale": k_scale,
        "v_packed": v_packed, "v_scale": v_scale,
        "D": D,
    }


def kv_dequant_unrotate(stored: dict, cfg: KVQConfig, out_dtype=torch.bfloat16):
    """KV-READ. Inverse of kv_rotate_quant: dequant -> inverse-rotate (transpose) ->
    out_dtype. Returns (k, v) ready for the SDPA (bf16 by default; the fp8-SDPA variant
    re-casts K to fp8 AFTER unrotation -- see note in the design header)."""
    D = stored["D"]
    R = make_rotation(D, use_signs=cfg.use_signs, seed=cfg.seed,
                      device=stored["k_fp8"].device)
    Rt = R.t().contiguous()                                     # inverse rotation
    k_rot = dequant_fp8_e4m3(stored["k_fp8"], stored["k_scale"])
    if cfg.v_bits == 4:
        v_codes = unpack_int4(stored["v_packed"], D)
    else:
        v_codes = unpack_int2(stored["v_packed"], D)
    v_rot = dequant_int_sym(v_codes, stored["v_scale"])
    k = (k_rot @ Rt).to(out_dtype)
    v = (v_rot @ Rt).to(out_dtype)
    return k, v


def reconstruct(k: torch.Tensor, v: torch.Tensor, cfg: KVQConfig):
    """Convenience: full round-trip K,V -> reconstructed K,V (same dtype as inputs)."""
    stored = kv_rotate_quant(k, v, cfg)
    return kv_dequant_unrotate(stored, cfg, out_dtype=k.dtype)


# =============================================================================
# fp8 KV baseline (the thing TurboQuant is measured against) -- plain per-(token,head)
# fp8 e4m3 on K AND V, NO rotation. This is the fp8_inc-equivalent control.
# =============================================================================
def fp8_kv_baseline(k: torch.Tensor, v: torch.Tensor, out_dtype=torch.bfloat16):
    """Per-(token,head) absmax fp8 e4m3 on both K and V, no rotation -- the fp8 control."""
    ks = _absmax_scale(k.float(), FP8_E4M3_MAX)
    vs = _absmax_scale(v.float(), FP8_E4M3_MAX)
    kq = (k.float() / ks).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    vq = (v.float() / vs).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return (kq.float() * ks).to(out_dtype), (vq.float() * vs).to(out_dtype)
