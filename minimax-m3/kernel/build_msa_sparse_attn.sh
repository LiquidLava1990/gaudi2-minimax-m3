#!/usr/bin/env bash
# =============================================================================
# build_msa_sparse_attn.sh -- MiniMax-M3 MSA: build msa_sparse_attn_backend.cpp
#   (EAGER) into build/libmsa_block_indexer.so
#
# This is the DSA dsa/build_dsa_indexer.sh adapted for the M3 MSA package, which
# lives OUTSIDE the opbackend tree ($HOME/AI/minimax-m3/kernel). The ONLY
# differences vs the DSA script:
#   - SRC/OUT point at this package's TU + build dir.
#   - BRIDGE is the EXISTING cloned bridge under hpu-moe-ext (READ-ONLY header use;
#     this script never writes to it except generating the one env_flags header it
#     already has). Override with $BRIDGE if you keep a separate clone.
#   - -I "${HERE}" so the TU finds its sibling msa_block_indexer_backend_shared.h.
# EVERYTHING ELSE (bridge headers, ABI=1, backend-lib-only link, the link-gap check)
# is byte-identical to the proven build.sh -- keep in lockstep.
#
# CPU-only HOST x86-64 compile (system g++). NO HPU device flags. Compiles C++
# against the cloned version-exact bridge headers (tag v1.24.0-1007) + torch 2.10 +
# synapse headers, and LINKS the SHIPPED libhabana_pytorch_backend.so (OpBackend::
# BuildNode, KernelRegistry, the GUID lowering). NEVER a plugin lib.
#
# Run (inside the dev image, NO device, NO card access):
#   docker run --rm \
#     -v $HOME/AI/minimax-m3:/m3 \
#     -v $HOME/AI/hpu-moe-ext:/hpumoe \
#     -e BRIDGE=/hpumoe/tpc_kernel/opbackend/bridge \
#     -w /m3/kernel vllm-gaudi:next bash build_msa_sparse_attn.sh
#
# Produces: build/libmsa_block_indexer.so  (loaded on the HPU box by the test).
# RUN THE OP ON THE HPU BOX ONLY (on-card validation is a later, gated step).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../minimax-m3/kernel
BUILD="${HERE}/build"
# BRIDGE: default to the existing cloned bridge under hpu-moe-ext (header-only use).
BRIDGE="${BRIDGE:-$HOME/AI/hpu-moe-ext/tpc_kernel/opbackend/bridge}"
mkdir -p "${BUILD}"

SRC="${HERE}/msa_sparse_attn_backend.cpp"
OUT="${BUILD}/libmsa_sparse_attn.so"
HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"

if [[ ! -d "${BRIDGE}" ]]; then
  echo "!!! BRIDGE='${BRIDGE}' missing. Point \$BRIDGE at the cloned gaudi-pytorch-bridge"
  echo "    (tag v1.24.0-1007), e.g. $HOME/AI/hpu-moe-ext/tpc_kernel/opbackend/bridge,"
  echo "    or clone it:  git clone --depth 1 --branch v1.24.0-1007 \\"
  echo "      https://github.com/HabanaAI/gaudi-pytorch-bridge.git \${BRIDGE}"
  exit 3
fi

# ---- build-only prerequisites (CPU headers only; identical to build.sh) ----
if [[ ! -f /usr/include/absl/functional/any_invocable.h ]]; then
  echo "=== installing libabsl-dev (build-only headers) ==="
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq libabsl-dev >/dev/null 2>&1 \
    || { echo "!!! could not apt-install libabsl-dev"; exit 3; }
fi

GEN_ENV_DIR="${BRIDGE}/generated/env_flags"
if [[ ! -f "${GEN_ENV_DIR}/env_flags_generated.h" ]]; then
  echo "=== generating env_flags_generated.h (scripts/generate_env.py) ==="
  mkdir -p "${GEN_ENV_DIR}"
  ( cd "${BRIDGE}/scripts" && \
    python3 generate_env.py -y env_flags.yaml -o "${GEN_ENV_DIR}" -p "${GEN_ENV_DIR}" ) \
    >/dev/null 2>&1 \
    || { echo "!!! env_flags generation failed"; exit 3; }
  [[ -f "${GEN_ENV_DIR}/env_flags_generated.h" ]] \
    || { echo "!!! env_flags_generated.h not produced"; exit 3; }
fi

# ---- torch + habana_frameworks discovery ----
TORCH_DIR="$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')"
HABF_DIR="$(python3 -c 'import habana_frameworks.torch as h,os;print(os.path.dirname(h.__file__))')"
TORCH_INC="${TORCH_DIR}/include"
TORCH_LIB="${TORCH_DIR}/lib"
HABF_INC="${HABF_DIR}/include"
HABF_LIB="${HABF_DIR}/lib"
PY_INC="$(python3 -c 'import sysconfig;print(sysconfig.get_path("include"))')"

ABI="$(python3 -c 'import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))' 2>/dev/null || echo 1)"

BACKEND_LIB_NAME="habana_pytorch_backend"
if [[ -f "${HABF_LIB}/libhabana_pytorch_backend.upstream.so" && \
      ! -f "${HABF_LIB}/libhabana_pytorch_backend.so" ]]; then
  BACKEND_LIB_NAME="habana_pytorch_backend.upstream"
fi

BRIDGE_INCS=(
  -I "${BRIDGE}"
  -I "${BRIDGE}/generated"
  -I "${BRIDGE}/pytorch_helpers"
  -I "${BRIDGE}/include"
  -I "${BRIDGE}/python_packages/habana_frameworks/torch/jit/csrc"
  -I "${BRIDGE}/third_party_vendored"
)

echo "=== build_msa_sparse_attn.sh (EAGER) ==="
echo "torch       : ${TORCH_DIR} (abi=${ABI})"
echo "habana_fw   : ${HABF_DIR}"
echo "backend lib : -l${BACKEND_LIB_NAME}  (plugins NOT linked; eager frontend syms runtime-resolved)"
echo "bridge      : ${BRIDGE}"
echo "src         : ${SRC}"
echo "out         : ${OUT}"
[[ -n "${EXTRA_CXXFLAGS}" ]] && echo "extra       : ${EXTRA_CXXFLAGS}"
echo

set -x
g++ -O2 -fPIC -shared -std=c++17 \
  -D_GLIBCXX_USE_CXX11_ABI="${ABI}" \
  -DHABANA_GAUDI2 \
  "${SRC}" \
  -I "${HERE}" \
  "${BRIDGE_INCS[@]}" \
  -I "${HAB_INC}" \
  -I "${TORCH_INC}" \
  -I "${TORCH_INC}/torch/csrc/api/include" \
  -I "${HABF_INC}" \
  -I "${PY_INC}" \
  ${EXTRA_CXXFLAGS} \
  -L "${TORCH_LIB}" -L "${HABF_LIB}" \
  -Wl,--as-needed \
  -ltorch -ltorch_cpu -lc10 \
  -l"${BACKEND_LIB_NAME}" \
  -Wl,--no-as-needed \
  -Wl,--allow-shlib-undefined \
  -Wl,-rpath,"${TORCH_LIB}" -Wl,-rpath,"${HABF_LIB}" \
  -o "${OUT}"
RC=$?
set +x

echo
if [[ ${RC} -eq 0 ]]; then
  echo "OK -> ${OUT}"
  echo
  echo "=== symbol resolution check ==="
  UNDEF="$(ldd -r "${OUT}" 2>&1 | grep -iE 'undefined symbol' || true)"
  UNDEF_SYMS="$(echo "${UNDEF}" | sed -E 's/[[:space:]]*\([^)]*\)[[:space:]]*$//')"
  LINK_GAP="$(echo "${UNDEF_SYMS}" | grep -E 'OpBackend|BuildNode|RegisterKernel|KernelRegistryEv|14KernelRegistry' || true)"
  if [[ -n "${LINK_GAP}" ]]; then
    echo "!!! UNRESOLVED backend symbols that the linked backend lib should provide (link gap):"
    echo "${LINK_GAP}" | head -20
    RC=4
  else
    echo "OK: OpBackend/BuildNode/KernelRegistry all resolve from the linked backend lib."
    EAGER_UNDEF="$(echo "${UNDEF_SYMS}" | grep -E 'habana::eager|EagerOp|HbEagerTensorPool' || true)"
    if [[ -n "${EAGER_UNDEF}" ]]; then
      n=$(echo "${EAGER_UNDEF}" | grep -c .)
      echo "Note: ${n} eager-frontend symbol(s) left UNDEFINED on purpose (resolved at runtime from the eager plugin)."
    fi
    echo "Demangled key symbols this .so defines / requires:"
    nm -C -D "${OUT}" 2>/dev/null | grep -iE 'sparse_attn|MsaSparseAttn' | head
  fi
  echo
  echo "On the HPU box (gated, later):  cd ${HERE} && python tests/test_msa_sparse_attn_op.py"
else
  echo "FAILED rc=${RC}"
fi
exit ${RC}
