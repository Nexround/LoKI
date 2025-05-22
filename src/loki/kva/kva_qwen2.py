from transformers import Qwen2ForCausalLM
import torch
import torch.nn.functional as F


class KVAQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self._intermediate_activations = []
        self._partitioning_activations = []
        self._partitioning_step = []
        self._partitioning_logits = []
        self.integrated_gradients = [None] * self.config.num_hidden_layers
        self._args = None
        self._kwargs = None
        self._new_dict = None

        for param in self.model.parameters():
            param.requires_grad = False
        for layer in self.model.layers:
            layer.mlp.down_proj.weight.requires_grad = True

    def forward(self, target_token_idx, *args, **kwargs):
        for layer in self.model.layers:
            layer.mlp.down_proj.weight.requires_grad = True
        self._args = args
        self._kwargs = kwargs
        keys_to_remove = ["input_ids", "attention_mask"]
        self._new_dict = {k: v for k, v in kwargs.items() if k not in keys_to_remove}

        # Hook to capture intermediate activations
        def hook_fn(module, input, output):
            self._intermediate_activations.append(output[:, target_token_idx, :])

        hooks = []
        for layer in self.model.layers:
            hooks.append(layer.mlp.down_proj.register_forward_hook(hook_fn))

        outputs = super().forward(*args, **kwargs).logits[:, target_token_idx, :]

        for hook in hooks:
            hook.remove()

        return outputs

    def forward_with_partitioning_single(
        self, target_token_idx, steps, predicted_label
    ):
        # Generate partitioned activations for all layers
        for param in self.model.parameters():
            param.requires_grad = False

        for vec in self._intermediate_activations:
            partitioning, step = self.generate_partitioning(vec, steps)
            self._partitioning_activations.append(partitioning)
            self._partitioning_step.append(step)

        num_layers = self.config.num_hidden_layers

        # Perform step-by-step inference
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].mlp.down_proj.weight.requires_grad = True

            layer_activation = self._partitioning_activations[layer_idx]
            layer_logits = []

            for i in range(steps):
                single_input_ids = self._kwargs["input_ids"]
                single_attention_mask = self._kwargs["attention_mask"]
                single_layer_activation = layer_activation[i].unsqueeze(0)

                # Register hook for a single sample
                hook = self._create_layer_hook(
                    target_token_idx=target_token_idx,
                    activations=single_layer_activation,
                    target=self.model.layers[layer_idx].mlp.down_proj,
                )

                outputs = self.model(
                    single_input_ids, single_attention_mask, **self._new_dict
                )
                single_logits = self.lm_head(
                    outputs.last_hidden_state[:, target_token_idx, :]
                )
                layer_logits.append(single_logits)
                prob = F.softmax(single_logits, dim=1)
                target_label_logits = prob[:, predicted_label]

                (gradient,) = torch.autograd.grad(
                    target_label_logits,
                    single_layer_activation,
                    grad_outputs=torch.ones_like(target_label_logits),
                )

                gradient = gradient.detach().cpu()

                hook.remove()
                with torch.no_grad():
                    self.integrated_gradients[layer_idx] = torch.zeros_like(
                        gradient.squeeze(0)
                    )
                    self.integrated_gradients[layer_idx] += (
                        gradient.squeeze(0) * self._partitioning_step[layer_idx]
                    )

            self.model.layers[layer_idx].mlp.down_proj.weight.requires_grad = False

        return self._partitioning_logits

    def forward_with_partitioning(self, target_token_idx, steps, predicted_label):
        # Generate partitioned activations for all layers
        for param in self.model.parameters():
            param.requires_grad = False

        for vec in self._intermediate_activations:
            partitioning, step = self.generate_partitioning(vec, steps)
            self._partitioning_activations.append(partitioning)
            self._partitioning_step.append(step)

        num_layers = self.config.num_hidden_layers

        # Perform batch inference layer-by-layer
        for layer_idx in range(num_layers):
            self.model.layers[layer_idx].mlp.down_proj.weight.requires_grad = True

            layer_input_ids = self._kwargs["input_ids"].repeat(steps, 1)
            layer_attention_mask = self._kwargs["attention_mask"].repeat(steps, 1)
            layer_activation = self._partitioning_activations[layer_idx]

            hook = self._create_layer_hook(
                target_token_idx=target_token_idx,
                activations=layer_activation,
                target=self.model.layers[layer_idx].mlp.down_proj,
            )
            outputs = self.model(
                layer_input_ids, layer_attention_mask, **self._new_dict
            )
            layer_logits = self.lm_head(
                outputs.last_hidden_state[:, target_token_idx, :]
            )

            self._partitioning_logits.append(layer_logits)
            hook.remove()
            self._compute_ig_for_layer(layer_idx, predicted_label)
            self.model.layers[layer_idx].mlp.down_proj.weight.requires_grad = False

        return self._partitioning_logits

    def reset_model(self):
        # Reset gradients
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
        torch.cuda.empty_cache()

    def _create_layer_hook(self, target_token_idx, activations, target):
        def hook_fn(module, input, output):
            output = output.clone()
            # Directly replace the target position for all samples in the batch
            output[:, target_token_idx] = activations
            return output

        return target.register_forward_hook(hook_fn)

    def generate_partitioning(self, vector, steps):
        baseline = torch.zeros_like(vector)
        step = (vector - baseline) / steps
        partitioning = torch.cat([baseline + step * i for i in range(steps)], dim=0)
        return partitioning, step[0].detach().cpu()

    def _compute_ig_for_layer(self, i, target_label):
        prob = F.softmax(self._partitioning_logits[i], dim=1)
        target_label_logits = prob[:, target_label]
        (gradient,) = torch.autograd.grad(
            target_label_logits,
            self._partitioning_activations[i],
            grad_outputs=torch.ones_like(target_label_logits),
        )
        gradient = gradient.detach().cpu()
        with torch.no_grad():
            self.integrated_gradients[i] = (
                gradient.sum(dim=0) * self._partitioning_step[i]
            )

    def clean(self):
        attrs = [
            "_intermediate_activations",
            "_partitioning_activations",
            "_partitioning_step",
            "_partitioning_logits",
            "integrated_gradients",
        ]
        for attr in attrs:
            getattr(self, attr).clear()
        self.integrated_gradients = [None] * self.config.num_hidden_layers
        self._args = None
        self._kwargs = None
