"""Serve-side EAGLE-3 support for MiniMax-M3 — IN-GRAPH aux (gated M3_EAGLE3=1).

The aux capture is now done IN the decoder forward (minimax_m3.py HpuMiniMaxM3Model.forward
collects the 3 low/mid/high layer outputs + returns (hidden, aux) when self.aux_hidden_state_layers
is set). That makes it part of the bridge's captured graph -> the EAGLE-3 head serves LAZY (the old
Python forward-hooks forced enforce-eager). This patch only adds the SupportsEagle3 interface so the
runner wires it, and points set_aux at the DECODER's attribute the in-graph forward reads.
Idempotent + byte-identical when M3_EAGLE3 unset."""
import os
import types

_LOG = lambda m: print(f"[m3-eagle3] {m}", flush=True)


def _find_decoder(model):
    """Walk to the object that holds `.layers` (the HpuMiniMaxM3Model decoder)."""
    for path in ("model.layers", "model.model.layers", "model.model.model.layers",
                 "language_model.model.layers"):
        obj = model
        try:
            for a in path.split(".")[:-1]:
                obj = getattr(obj, a)
            if hasattr(obj, "layers") and hasattr(obj.layers, "__len__") and len(obj.layers) > 4:
                return obj
        except Exception:  # noqa: BLE001
            continue
    return None


def install_eagle3_support(model) -> bool:
    if os.environ.get("M3_EAGLE3", "0") not in ("1", "true", "True"):
        return False
    if getattr(model, "_eagle3_patched", False):
        return True
    decoder = _find_decoder(model)
    if decoder is None:
        _LOG("could NOT find decoder (.layers holder) -> eagle3 NOT installed")
        return False

    L = len(decoder.layers)
    default_aux = (max(0, L // 4), L // 2, L - 1)   # (15,30,59) for 60 -> matches training capture

    # SupportsEagle3 protocol members (instance attrs satisfy runtime_checkable isinstance)
    model.supports_eagle3 = True
    model.has_own_embed_tokens = True
    model.has_own_lm_head = True

    def _set_aux(self, layers):
        # point the IN-GRAPH decoder forward at these layers (it reads decoder.aux_hidden_state_layers)
        decoder.aux_hidden_state_layers = tuple(int(x) for x in layers)
        _LOG(f"aux_hidden_state_layers set on decoder -> {decoder.aux_hidden_state_layers} "
             f"(IN-GRAPH capture; lazy-servable)")

    def _get_default(self):
        return default_aux

    model.set_aux_hidden_state_layers = types.MethodType(_set_aux, model)
    model.get_eagle3_default_aux_hidden_state_layers = types.MethodType(_get_default, model)
    model.get_eagle3_aux_hidden_state_layers = types.MethodType(_get_default, model)
    model._eagle3_patched = True
    _LOG(f"EAGLE-3 IN-GRAPH support INSTALLED (decoder L={L}, default aux {default_aux}; "
         f"no Python hooks -> serves LAZY).")
    return True
