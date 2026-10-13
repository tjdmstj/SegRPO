"""Point CUDA_HOME at a real toolkit before anything imports transformers.

~/.bashrc line 121 exports CUDA_HOME=/usr/local/cuda-13.0, and that directory does not
exist on this machine. transformers imports deepspeed, deepspeed's op builder runs
$CUDA_HOME/bin/nvcc at IMPORT time, and the whole thing dies with

    RuntimeError: Failed to import transformers.models.whisper.modeling_whisper ...
    [Errno 2] No such file or directory: '/usr/local/cuda-13.0/bin/nvcc'

which names neither the real cause nor the file that set it. Clearing the variable does
not help either: deepspeed then raises MissingCUDAException. The only nvcc on this box
lives inside a few conda envs, so point at one of those. Nothing here ever builds a
deepspeed op, so the toolkit is asked for nothing but its version string.

PATH is deliberately left alone. Prepending a conda env's bin/ shadows the ACTIVE env's
python, which surfaces later as an unrelated-looking ModuleNotFoundError.

    import cuda_home  # noqa: F401   <- before transformers
"""
import glob
import os

CANDIDATES = ['/usr/local/cuda',
              '',
              '',
              '']


def ensure(verbose=False):
    cur = os.environ.get('CUDA_HOME')
    if cur and os.path.exists(os.path.join(cur, 'bin', 'nvcc')):
        return cur
    for c in CANDIDATES + sorted(glob.glob('')):
        if os.path.exists(os.path.join(c, 'bin', 'nvcc')):
            os.environ['CUDA_HOME'] = c
            if verbose:
                print(f'[cuda_home] {cur or "unset"} -> {c}')
            return c
    return cur


ensure()
