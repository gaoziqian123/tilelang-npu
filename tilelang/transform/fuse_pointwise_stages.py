"""Conservative adjacent private memory-version forwarding (native pass)."""
from . import _ffi_api


def FusePointwiseStages():
    """Fuse same-domain pointwise stages whose consumer kills the private version.

    No reassociation or cast elimination is performed. Unsupported effects,
    views, indices and control flow retain the original IR.
    """
    return _ffi_api.FusePointwiseStages()
