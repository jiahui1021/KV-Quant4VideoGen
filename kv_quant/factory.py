from __future__ import annotations

import re


SUPPORTED_METHODS = (
    "BF16",
    "RTN_INT4",
    "RTN_INT2",
    "KIVI_INT4",
    "KIVI_INT2",
    "QUAROT_KV_INT4",
    "QUAROT_KV_INT2",
    "HADAMARD_K_INT4",
    "HADAMARD_K_INT2",
    "QVG_INT2",
    "QVG_INT4",
)


def create_quantizer(method: str, bits: int, block_size: int | None = None, **options):
    """Create one of the shared KV-cache quantizers.

    ``block_size=None`` keeps the shared comparison defaults: groups of 64,
    with a 128-token BF16 residual for KIVI.  A number overrides the RTN
    channel group and the KIVI key group; QuaRot's group is set by
    ``channel_group_size`` alone.  ``options`` go to the constructor, which
    rejects any knob the method does not have (e.g. ``asym`` for RTN) rather
    than silently ignoring it.
    """
    method = method.upper()
    options = {name: value for name, value in options.items() if value is not None}
    if method == "RTN":
        from .rtn import RTNQuantizer as cls
    elif method == "KIVI":
        from .kivi import KIVIQuantizer as cls
    elif method == "QUAROT_KV":
        from .quarot_kv import QuaRotKVQuantizer as cls
    elif method == "HADAMARD_K":
        from .quarot_kv import HadamardKQuantizer as cls
    else:
        raise ValueError(f"Unsupported KV quantization method: {method}")
    if block_size is not None and method in ("RTN", "KIVI"):
        options["block_size"] = int(block_size)
    return cls(bits=bits, **options)


def parse_method(
    method: str,
    block_size: int | None = None,
    channel_group_size: int | None = None,
    asym: bool | None = None,
    clip_ratio: float | None = None,
):
    """Return ``(canonical_name, quantizer_or_none)`` for a CLI method.

    ``asym`` and ``clip_ratio`` are QuaRot's ``--k_asym``/``--k_clip_ratio``
    knobs and only apply to the rotated backends; passing them to RTN or KIVI
    is refused by the constructor rather than ignored.
    """
    method = method.upper()
    if method == "BF16":
        return "BF16", None

    match = re.fullmatch(r"(RTN|KIVI|QUAROT_KV|HADAMARD_K)_INT(2|4)", method)
    if match is None:
        supported = ", ".join(
            name for name in SUPPORTED_METHODS if not name.startswith("QVG_")
        )
        raise ValueError(
            f"Unsupported method={method}. Expected one of: {supported} "
            "(QVG has its own entry point in the Causal-Forcing adapter)"
        )

    return method, create_quantizer(
        match.group(1),
        int(match.group(2)),
        block_size=block_size,
        channel_group_size=channel_group_size,
        asym=asym,
        clip_ratio=clip_ratio,
    )
