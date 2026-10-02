"""Source-only adapter for externally compiled and deployed kernels."""

from .base import BaseKernelAdapter


class AOTKernelAdapter(BaseKernelAdapter):
    def __init__(self, artifact, result_idx):
        self.kernel_source = artifact.kernel_source
        super().__init__(artifact.device_mod, artifact.params, result_idx)

    def _convert_torch_func(self):
        def external_execution_required(*args, **kwargs):
            raise RuntimeError("AOT kernels must be compiled and deployed by the external device runtime")

        return external_execution_required

    def get_kernel_source(self, kernel_only=True):
        return self.kernel_source
