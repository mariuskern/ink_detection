"""
Helpers for parsing Lightning-style precision strings into native
PyTorch precision settings.
"""

from contextlib import nullcontext

import torch


def parse_precision(precision_str: str):
    """
    Convert a Lightning precision string to native PyTorch settings.

    Returns
    -------
    autocast_dtype : torch.dtype | None
        Dtype for torch.amp.autocast.

    use_grad_scaler : bool
        Whether to use torch.amp.GradScaler.

    model_dtype : torch.dtype | None
        Dtype to convert the model to for ``*-true`` precision.
        None means that the model should remain in its default dtype.

    Supported:
        32-true
        16-mixed
        16-true
        bf16-mixed
        bf16-true
    """

    mapping = {
        # Full FP32 training
        "32-true": (
            None,          # autocast_dtype
            False,         # use_grad_scaler
            None,          # model_dtype
        ),

        # AMP FP16: model remains FP32
        "16-mixed": (
            torch.float16,
            True,
            None,
        ),

        # Full FP16 model
        "16-true": (
            None,
            False,
            torch.float16,
        ),

        # AMP BF16: model remains FP32
        "bf16-mixed": (
            torch.bfloat16,
            False,
            None,
        ),

        # Full BF16 model
        "bf16-true": (
            None,
            False,
            torch.bfloat16,
        ),
    }

    if precision_str not in mapping:
        raise ValueError(f"Unknown precision string: {precision_str!r}. Supported: {list(mapping.keys())}")

    return mapping[precision_str]


def autocast_context(device_type: str, autocast_dtype):
    """
    Return an autocast context manager, or a no-op if autocast_dtype is None.
    """

    if autocast_dtype is None:
        return nullcontext()

    return torch.amp.autocast(device_type=device_type, dtype=autocast_dtype)



# """
# Helpers for parsing Lightning-style precision strings into native
# ``torch.amp`` arguments.
# """

# from contextlib import nullcontext

# import torch


# def parse_precision(precision_str: str):
#     """
#     Convert a Lightning precision string to ``(autocast_dtype, use_grad_scaler)``.

#     Supported strings (same as Lightning):

#     +--------------+-------------------+------------------+
#     | String       | autocast dtype    | GradScaler       |
#     +==============+===================+==================+
#     | ``32-true``  | None (no AMP)     | False            |
#     | ``16-mixed`` | ``float16``       | True             |
#     | ``bf16-mixed``| ``bfloat16``     | False            |
#     | ``16-true``  | ``float16``       | False            |
#     | ``bf16-true``| ``bfloat16``      | False            |
#     +--------------+-------------------+------------------+

#     Returns
#     -------
#     autocast_dtype : torch.dtype | None
#         The dtype for ``torch.amp.autocast``, or ``None`` for full precision.
#     use_grad_scaler : bool
#         Whether to use ``torch.amp.GradScaler``.
#     """
#     mapping = {
#         "32-true":    (None,              False),
#         "16-mixed":   (torch.float16,     True),
#         "16-true":    (torch.float16,     False),
#         "bf16-mixed": (torch.bfloat16,    False),
#         "bf16-true":  (torch.bfloat16,    False),
#     }

#     if precision_str not in mapping:
#         raise ValueError(f"Unknown precision string: {precision_str!r}. Supported: {list(mapping.keys())}")

#     return mapping[precision_str]


# def autocast_context(device_type: str, autocast_dtype):
#     """
#     Return an ``autocast`` context manager, or a no-op if *autocast_dtype*
#     is ``None`` (i.e. full-precision training).
#     """
#     if autocast_dtype is None:
#         return nullcontext()

#     return torch.amp.autocast(device_type=device_type, dtype=autocast_dtype)
