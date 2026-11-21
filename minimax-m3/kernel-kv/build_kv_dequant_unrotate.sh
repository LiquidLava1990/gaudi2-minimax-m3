#!/usr/bin/env bash
# =============================================================================
# build_kv_dequant_unrotate.sh -- MiniMax-M3 TurboQuant KV-READ: build
#   kv_dequant_unrotate_backend.cpp (EAGER) into build/libkv_dequant_unrotate.so
#
# Byte-identical to the proven build_msa_block_indexer.sh EXCEPT SRC/OUT and the
# package dir ($HOME/AI/minimax-m3/kernel-kv). SEPARATE package, own .so, own
# namespace (hpu_kvq_ext); never rebuilds the installed hpu-moe-ext/hpu-dsa-ext .so
# and never writes to /opt. Uses the EXISTING cloned bridge headers READ-ONLY.
#
# CPU-only HOST x86-64 compile (system g++). NO HPU device flags, NO card. Compiles
# against the version-exact bridge headers (tag v1.24.0-1007) + torch + synapse and
# LINKS the SHIPPED libhabana_pytorch_backend.so (OpBackend::BuildNode, KernelRegistry,
# GUID lowering). NEVER a plugin lib.
#
# Run (inside the dev image, NO device, NO card access):
#   docker run --rm \
#     -v $HOME/AI/minimax-m3:/m3 \
#     -v $HOME/AI/hpu-moe-ext:/hpumoe \
#     -e BRIDGE=/hpumoe/tpc_kernel/opbackend/bridge \
#     -w /m3/kernel-kv vllm-gaudi:next bash build_kv_dequant_unrotate.sh
#
# Produces: build/libkv_dequant_unrotate.so. RUN THE OP ON THE HPU BOX ONLY (gated).
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../minimax-m3/kernel-kv
BUILD="${HERE}/build"
BRIDGE="${BRIDGE:-$HOME/AI/hpu-moe-ext/tpc_kernel/opbackend/bridge}"
mkdir -p "${BUILD}"

SRC="${HERE}/kv_dequant_unrotate_backend.cpp"
OUT="${BUILD}/libkv_dequant_unrotate.so"
HAB_INC="${HAB_INC:-/usr/include/habanalabs}"
EXTRA_CXXFLAGS="${EXTRA_CXXFLAGS:-}"

if [[ ! -d "${BRIDGE}" ]]; then
  echo "!!! BRIDGE='${BRIDGE}' missing. Point \$BRIDGE at the cloned gaudi-pytorch-bridge"
  echo "    (tag v1.24.0-1007), e.g. $HOME/AI/hpu-moe-ext/tpc_kernel/opbackend/bridge."
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

echo "=== build_kv_dequant_unrotate.sh (EAGER) ==="
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
    echo "!!! UNRESOLVED backend symbols the linked backend lib should provide (link gap):"
    echo "${LINK_GAP}" | head -20
    RC=4
  else
    echo "OK: OpBackend/BuildNode/KernelRegistry all resolve from the linked backend lib."
    EAGER_UNDEF="$(echo "${UNDEF_SYMS}" | grep -E 'habana::eager|EagerOp|HbEagerTensorPool' || true)"
    if [[ -n "${EAGER_UNDEF}" ]]; then
      n=$(echo "${EAGER_UNDEF}" | grep -c .)
      echo "Note: ${n} eager-frontend symbol(s) left UNDEFINED on purpose (runtime-resolved from the eager plugin)."
    fi
    echo "Demangled key symbols this .so defines / requires:"
    nm -C -D "${OUT}" 2>/dev/null | grep -iE 'kv_dequant_unrotate|KvDequantUnrotate' | head
  fi
  echo
  echo "On the HPU box (gated, later):  cd ${HERE} && python tests/test_turboquant_kv_op.py"
else
  echo "FAILED rc=${RC}"
fi
exit ${RC}
