from tilelang.backend.execution_backend import ExecutionBackendSpec

# Produce source only. No host LLVM module or local DSP runtime is required.
EXECUTION_BACKENDS = (ExecutionBackendSpec("aot"),)
