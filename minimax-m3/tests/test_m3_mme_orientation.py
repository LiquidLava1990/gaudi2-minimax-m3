#!/usr/bin/env python3
"""Card-free regression test for MiniMax-M3 fp8_gemm_v2 weight orientation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace


class ShapeOnly:
    def __init__(self, *shape: int) -> None:
        self.shape = shape

    def reshape(self, *_shape: int) -> "ShapeOnly":
        return self


def load_module() -> ModuleType:
    source = Path(__file__).with_name("m3_expert_runtime_patch.py")
    if not source.exists():
        source = Path(__file__).parent.parent / "m3_expert_runtime_patch.py"
    spec = importlib.util.spec_from_file_location("m3_expert_runtime_patch_under_test", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_layout(module: ModuleType, op: SimpleNamespace, expected_transpose: bool) -> None:
    _w13, _w2, transpose, layout = module._mme_handles(op)
    assert transpose is expected_transpose, layout


def main() -> None:
    module = load_module()
    out_in = SimpleNamespace(
        _w13_oi_stack="w13-out-in",
        _w2_oi_stack="w2-out-in",
        _w13_fp8_stack="w13-in-out",
        _w2_fp8_stack="w2-in-out",
    )
    assert_layout(module, out_in, True)
    module._mme_shape_ok[0] = False
    module._assert_shapes_mme(
        None, ShapeOnly(1, 6144), ShapeOnly(6144, 6144), ShapeOnly(6144, 3072),
        ShapeOnly(6144), ShapeOnly(6144), True, 4, "[out,in]",
    )

    in_out = SimpleNamespace(
        _w13_oi_stack=None,
        _w2_oi_stack=None,
        _w13_fp8_stack="w13-in-out",
        _w2_fp8_stack="w2-in-out",
    )
    assert_layout(module, in_out, False)
    module._mme_shape_ok[0] = False
    module._assert_shapes_mme(
        None, ShapeOnly(1, 6144), ShapeOnly(6144, 6144), ShapeOnly(3072, 6144),
        ShapeOnly(6144), ShapeOnly(6144), False, 4, "[in,out]",
    )

    module._mme_shape_ok[0] = False
    try:
        module._assert_shapes_mme(
            None, ShapeOnly(1, 6144), ShapeOnly(6144, 6144), ShapeOnly(6144, 3072),
            ShapeOnly(6144), ShapeOnly(6144), False, 4, "invalid old convention",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("the old reversed trans_B convention was not rejected")


if __name__ == "__main__":
    main()
