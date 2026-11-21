#!/usr/bin/env bash
# =============================================================================
# build_msa_sparse_attn_lazy.sh -- MiniMax-M3 MSA: build
#   msa_sparse_attn_backend_lazy.cpp into build/libmsa_sparse_attn_lazy.so
#   (the LAZY-FRONTEND variant -- the production serving path).
#
# DSA dsa/build_dsa_indexer_lazy.sh adapted for the M3 MSA package (outside the
# opbackend tree). Discovery python + symbol checks run UNDER PT_HPU_LAZY_MODE=1 so
# the LAZY plugin (libhabana_pytorch_plugin.so) is referenced -- the plugin that
# resolves this .so's habana_lazy::* frontend symbols at runtime. Links ONLY the
# shipped libhabana_pytorch_backend.so (NEVER a plugin).
#
# Run (inside the dev image, NO device):
#   docker run --rm \
#     -v $HOME/AI/minimax-m3:/m3 -v $HOME/AI/hpu-moe-ext:/hpumoe \
#     -e BRIDGE=/hpumoe/tpc_kernel/opbackend/bridge \
#     -w /m3/kernel vllm-gaudi:next bash build_msa_sparse_attn_lazy.sh
#
# Produces: build/libmsa_block_indexer_lazy.so. RUN ON THE HPU BOX ONLY (gated).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="${HERE}/build"
BRIDGE="${BRIDGE:-$HOME/AI/hpu-moe-ext/tpc_kernel/opbackend/bridge}"
mkdir -p "${BUILD}"

SRC="${HERE}/msa_sparse_attn_backend_lazy.cpp"
OUT="${BUILD}/libmsa_sparse_attn_lazy.so"
HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"

# Reference the LAZY plugin for discovery + symbol checks.
export PT_HPU_LAZY_MODE=1

if [[ ! -d "${BRIDGE}" ]]; then
  echo "!!! BRIDGE='${BRIDGE}' missing. Point \$BRIDGE at the cloned gaudi-pytorch-bridge (tag v1.24.0-1007)."
  exit 3
fi

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

echo "=== build_msa_sparse_attn_lazy.sh (LAZY-frontend variant) ==="
echo "PT_HPU_LAZY_MODE : ${PT_HPU_LAZY_MODE} (discovery + symbol checks reference the LAZY plugin)"
echo "torch       : ${TORCH_DIR} (abi=${ABI})"
echo "habana_fw   : ${HABF_DIR}"
echo "backend lib : -l${BACKEND_LIB_NAME}  (plugins NOT linked; habana_lazy::* syms runtime-resolved)"
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
    LAZY_UNDEF="$(echo "${UNDEF_SYMS}" | grep -E 'habana_lazy|LazyOp|flush_op|AccThread' || true)"
    if [[ -n "${LAZY_UNDEF}" ]]; then
      n=$(echo "${LAZY_UNDEF}" | grep -c .)
      echo "Note: ${n} lazy-frontend symbol(s) left UNDEFINED on purpose (resolved at runtime from the lazy plugin)."
    fi
    echo "Demangled key symbols this .so defines / requires:"
    nm -C -D "${OUT}" 2>/dev/null | grep -iE 'sparse_attn|MsaSparseAttn' | head
  fi
  echo
  echo "On the HPU box (lazy, gated):  cd ${HERE} && PT_HPU_LAZY_MODE=1 MSA_SPARSE_ATTN_SO=${OUT} python tests/test_msa_sparse_attn_op.py"
else
  echo "FAILED rc=${RC}"
fi
exit ${RC}
