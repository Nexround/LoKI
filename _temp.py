import json
from pathlib import Path
from typing import Type, Union
from glob import glob

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
)
from safetensors import safe_open

from .loki_linear import LoKILinear


def create_loki_model(
    loki_model_class: Type[PreTrainedModel],
    loki_config_cls: Type[PretrainedConfig],
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    target_pos_path: Union[str, Path] = "",
    save_dir: Union[str, Path] = "./loki_model",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Create and save a LoKI model by loading the original model, applying
    position indices, and saving the modified weights and tokenizer.

    Args:
        loki_model_class: The LoKI model class to load.
        loki_config_cls: Corresponding LoKI configuration class.
        model_name: Identifier of the base pretrained model.
        target_pos_path: Path to JSON file specifying position indices to modify.
        save_dir: Directory where the resulting model and tokenizer will be saved.
        torch_dtype: Data type for model weights (e.g., torch.bfloat16).
    """
    # Load position indices from JSON file
    with open(target_pos_path, "r", encoding="utf-8") as f:
        target_pos = json.load(f)

    # Load the original pretrained model
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype
    )

    # Register LoKI classes for auto mapping
    loki_model_class.register_for_auto_class("AutoModelForCausalLM")
    loki_config_cls.register_for_auto_class()

    # Initialize LoKI configuration with specified position indices
    loki_config = loki_config_cls.from_pretrained(model_name, target_pos=target_pos)

    # Load the LoKI model using the original pretrained weights and new config
    loki_model = loki_model_class.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=loki_config,
        torch_dtype=torch_dtype,
    )

    # This step will save the configuration file of the LoKI model so that any code that supports the Transformers library can use the LoKI model
    loki_model.save_pretrained(save_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_dir)

    # Please note that this step is necessary as the LoKI Linear is instantly created every time the LoKI model is loaded
    # So we need the weights of the original model
    original_model.save_pretrained(save_dir, is_main_process=False)

    print("Successfully created and saved LoKI model.")


def _merge_loki_weights(
    loki_layer: LoKILinear,
    original_linear: torch.nn.Linear,
) -> None:
    """
    Merge weights and biases from a LoKI linear layer back into the original
    linear layer.

    Args:
        loki_layer: The LoKI-wrapped linear layer with active and frozen parts.
        original_linear: The corresponding linear layer from the original model.
    """
    # Allocate space for the merged weight matrix
    merged_weight = torch.zeros_like(original_linear.weight.data)

    # Insert active and frozen weights into correct positions
    merged_weight[loki_layer.active_pos] = loki_layer.active.weight.data
    merged_weight[loki_layer.frozen_pos] = loki_layer.frozen.weight.data

    # If bias exists, merge it similarly
    if original_linear.bias is not None:
        merged_bias = torch.zeros_like(original_linear.bias.data)
        merged_bias[loki_layer.active_pos] = loki_layer.active_bias.data
        merged_bias[loki_layer.frozen_pos] = loki_layer.frozen_bias.data

    # Overwrite the original layer's parameters
    original_linear.weight.data.copy_(merged_weight)
    if original_linear.bias is not None:
        original_linear.bias.data.copy_(merged_bias)


def set_zero_weights(
    target_pos_path: Union[str, Path],
    output_path: Union[str, Path],
    model_name: Union[str, Path],
    torch_dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    Zero out the weights of specified weights in the down-projection layers
    of an original pretrained model, and save the modified model.

    Args:
        target_pos_path: JSON path listing position indices per layer to zero.
        output_path: Directory to save the modified model and tokenizer.
        model_name: Base model identifier to load original weights.
        torch_dtype: Data type for loading the model.
    """
    # Load position indices
    with open(target_pos_path, "r", encoding="utf-8") as f:
        target_pos = json.load(f)

    # Load the original pretrained model
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype
    )

    # Iterate over all transformer layers
    for layer_idx in range(original_model.config.num_hidden_layers):
        down_proj = original_model.model.layers[layer_idx].mlp.down_proj

        # Zero out weights for each specified position index
        with torch.no_grad():
            for idx in target_pos.get(str(layer_idx), []):
                if 0 <= idx < down_proj.weight.size(0):
                    down_proj.weight[idx, :] = 0.0

    # Save the modified model and tokenizer
    original_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)

def restore_loki_model(
    target_pos_path: Union[str, Path],
    model_path: Union[str, Path],
    model_name: Union[str, Path],
    output_path: Union[str, Path],
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Restores LoKI modified weights into the base model architecture.
    
    Args:
        target_pos_path: Path to JSON file containing target position indices
        model_path: Directory containing LoKI modified weights
        model_name: Name/path of the base model to restore
        output_path: Destination path for saved restored model
        torch_dtype: Torch data type for model loading
    """
    
    # Load target position indices from JSON file
    with open(target_pos_path, "r", encoding="utf-8") as f:
        target_pos = json.load(f)

    # Load base transformer model
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype
    )

    # Initialize tensor dictionary for modified weights
    tensor_dict = {}
    
    # Handle single safetensors file case
    safe_tensor_path = Path(model_path) / "model.safetensors"
    if safe_tensor_path.is_file():
        # Load from single safetensors file
        with safe_open(safe_tensor_path, framework="pt") as f:
            tensor_dict.update({key: f.get_tensor(key) for key in f.keys()})
    else:
        # Load from sharded safetensors files
        shard_files = sorted(glob(f"{model_path}/model-*-of-*.safetensors"))
        for shard_file in shard_files:
            with safe_open(shard_file, framework="pt") as f:
                tensor_dict.update({key: f.get_tensor(key) for key in f.keys()})

    # Process each transformer layer
    for layer_idx in range(original_model.config.num_hidden_layers):
        # Skip layers without modifications
        if not target_pos[layer_idx]:
            continue

        # Get original down projection layer
        original_down_proj = original_model.model.layers[layer_idx].mlp.down_proj
        
        # Initialize LoKI modified layer
        loki_layer = LoKILinear(
            original_down_proj, target_pos=target_pos[layer_idx]
        )
        
        # Prepare state dictionary for weight loading
        state_dict = {}
        
        # Load weight parameters
        weight_keys = {
            "active.weight": f"model.layers.{layer_idx}.mlp.down_proj.active.weight",
            "frozen.weight": f"model.layers.{layer_idx}.mlp.down_proj.frozen.weight",
        }
        for param_key, tensor_key in weight_keys.items():
            if tensor_key in tensor_dict:
                state_dict[param_key] = tensor_dict[tensor_key]

        # Load bias parameters if present in original layer
        if original_down_proj.bias is not None:
            bias_keys = {
                "active_bias": f"model.layers.{layer_idx}.mlp.down_proj.active_bias",
                "frozen_bias": f"model.layers.{layer_idx}.mlp.down_proj.frozen_bias"
            }
            for param_key, tensor_key in bias_keys.items():
                if tensor_key in tensor_dict:
                    state_dict[param_key] = tensor_dict[tensor_key]

        # Load parameters into LoKI layer
        loki_layer.load_state_dict(state_dict, strict=False)

        # Integrate LoKI parameters back into base model
        _merge_loki_weights(loki_layer, original_down_proj)

    # Save reconstructed model and tokenizer
    original_model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_path)

    print(f"Model restored and saved to {output_path}")