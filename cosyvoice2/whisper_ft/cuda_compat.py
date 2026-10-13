"""Repair CUDA_HOME before transformers is imported.

~/.bashrc line 121 exports CUDA_HOME=/usr/local/cuda-13.0, and that directory does not
exist on this machine. Any process that imports transformers in an env where deepspeed is
installed (the `cosyvoice` env is) then dies at import time, because deepspeed's op builder
shells out to $CUDA_HOME/bin/nvcc just to read a version string:

    RuntimeError: Failed to import transformers.models.whisper.modeling_whisper ...
    [Errno 2] No such file or directory: '/usr/local/cuda-13.0/bin/nvcc'

Unsetting CUDA_HOME is not a fix either -- deepspeed then raises MissingCUDAException.
nvcc does exist inside several conda envs here, so point at one. Nothing in whisper_ft
ever builds a CUDA op, so the toolkit is only ever asked for its version.

Import this FIRST, before transformers:

    import cuda_compat  # noqa: F401
    from transformers import ...
"""
import os

CANDIDATES = (
    '/usr/local/cuda',
    '',        # 12.4, closest to torch cu121
    '',     # 12.9
    '',        # 13.0
)


def _ok(home):
    return bool(home) and os.path.exists(os.path.join(home, 'bin', 'nvcc'))


def fix():
    """-> the CUDA_HOME now in force, or None if deepspeed is absent / none was found."""
    if _ok(os.environ.get('CUDA_HOME')):
        return os.environ['CUDA_HOME']
    import importlib.util
    if importlib.util.find_spec('deepspeed') is None:
        # nothing in this env will look for nvcc; leave the environment alone
        return None
    for c in CANDIDATES:
        if _ok(c):
            os.environ['CUDA_HOME'] = c
            return c
    return None


fix()
