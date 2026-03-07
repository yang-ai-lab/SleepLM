import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from .biosignals_coca_model import BiosignalsCoCa
from .model import get_cast_dtype, convert_weights_to_lp
from .tokenizer import SimpleTokenizer, DEFAULT_CONTEXT_LENGTH

_MODEL_CONFIG_PATHS = [Path(__file__).parent / "model_configs/"]
_MODEL_CONFIGS = {}


def _rescan_model_configs():
    global _MODEL_CONFIGS
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_dir():
            config_files.extend(config_path.glob("*.json"))
    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "biosignals_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg


_rescan_model_configs()


def get_model_config(model_name: str):
    return deepcopy(_MODEL_CONFIGS.get(model_name))


def create_model(
    model_name: str,
    precision: str = "fp32",
    device: Union[str, torch.device] = "cpu",
    **model_kwargs,
) -> BiosignalsCoCa:
    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = get_model_config(model_name)
    if model_cfg is None:
        raise RuntimeError(f"Model config for '{model_name}' not found. Available: {list(_MODEL_CONFIGS.keys())}")

    model_cfg.pop("custom_text", None)
    model_cfg.update(model_kwargs)

    cast_dtype = get_cast_dtype(precision)
    model = BiosignalsCoCa(**model_cfg, cast_dtype=cast_dtype)

    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if "fp16" in precision else torch.bfloat16
        model.to(device=device)
        convert_weights_to_lp(model, dtype=dtype)
    elif precision in ("pure_fp16", "pure_bf16"):
        dtype = torch.float16 if "fp16" in precision else torch.bfloat16
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)

    model.output_dict = True
    return model


def load_checkpoint(model, checkpoint_path: str, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if next(iter(state_dict)).startswith("module."):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    incompatible = model.load_state_dict(state_dict, strict=False)
    return incompatible


def get_tokenizer(model_name: str = "", context_length: Optional[int] = None, **kwargs):
    config = get_model_config(model_name) or {}
    text_cfg = config.get("text_cfg", {})
    if context_length is None:
        context_length = text_cfg.get("context_length", DEFAULT_CONTEXT_LENGTH)
    return SimpleTokenizer(context_length=context_length, **kwargs)


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ("bf16", "pure_bf16"):
        input_dtype = torch.bfloat16
    elif precision in ("fp16", "pure_fp16"):
        input_dtype = torch.float16
    return input_dtype
