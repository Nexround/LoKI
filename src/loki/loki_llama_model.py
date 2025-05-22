from transformers import LlamaForCausalLM

from .loki_llama_config import LoKILlamaConfig
from .loki_linear import LoKILinear


class LoKILlamaForCausalLM(LlamaForCausalLM):
    config_class = LoKILlamaConfig

    def __init__(self, config):
        if not hasattr(config, 'target_pos') or config.target_pos is None:
            raise ValueError(
                f"Config must include `target_pos`, but got: {config}"
            )
        super().__init__(config)  # Initialize parent model

        self.target_pos = config.target_pos
        # Validate neuron configuration matches number of layers
        if len(self.target_pos) != config.num_hidden_layers:
            raise ValueError(
                f"Length of target_pos ({len(self.target_pos)}) must equal num_hidden_layers ({config.num_hidden_layers})"
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Automatically load corresponding config file
        config = kwargs.pop('config', None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
            
        # Call parent class to load pretrained model
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            *args,
            config=config,
            **kwargs
        )
        # Freeze base model parameters
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = False
        
        # Replace target linear layers
        model.apply_loki_linear()
        
        return model

    def apply_loki_linear(self):
        """Replace all target MLP projection layers with LoKILinear"""
        if not hasattr(self, 'config') or not hasattr(self, 'model'):
            raise ValueError
        for layer_idx in range(self.config.num_hidden_layers):
            mlp_layer = self.model.layers[layer_idx].mlp
            original_layer = mlp_layer.down_proj
            target_pos = self.target_pos[layer_idx]
            
            if len(target_pos) == 0:
                continue  # Skip layers with no target neurons
                
            # Initialize LoKI layer and replace
            loki_linear = LoKILinear(
                original_linear=original_layer,
                target_pos=target_pos
            )
            
            setattr(mlp_layer, 'down_proj', loki_linear)
            print(f"Replaced down_proj in layer {layer_idx}")
