"""Neutralise peft's torchao probe on this machine.

peft 0.20 asks is_torchao_available() for EVERY module it wraps, and that helper does not
return False for an old torchao -- it raises:

    ImportError: Found an incompatible version of torchao. Found version 0.9.0,
                 but only versions above 0.16.0 are supported

so get_peft_model() dies before a single LoRA layer is created, even though nothing here
uses torchao (the base model is plain bf16, not quantised). Patching the probe to False
makes peft skip the torchao dispatcher and fall through to the ordinary nn.Linear one,
which is the path this recipe wants anyway.

`pip install -U torchao` fixes it properly, but this env is shared with other projects, so
the shim is the smaller change. Import this before peft is used.
"""
import peft.import_utils

peft.import_utils.is_torchao_available = lambda: False
try:
    import peft.tuners.lora.torchao as _lora_torchao      # imports the symbol by name

    _lora_torchao.is_torchao_available = lambda: False
except ImportError:                                        # pragma: no cover
    pass
