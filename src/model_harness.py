"""
Model Harness & Residual Stream Activation Hook Engine.
Provides:
1. Device-aware model loading (Apple Silicon MPS / CUDA / CPU)
2. Non-invasive PyTorch forward hooks to record residual stream activations
3. Causal activation steering / ablation injection hooks
4. Robust parsing for reasoning models (extracting <think>...</think> CoT and final answers)
"""

import re
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

class ActivationCache:
    """Stores activations across layers and sequence token positions."""
    def __init__(self):
        self.activations: Dict[int, List[torch.Tensor]] = {} # layer_idx -> list of [batch, seq_len, hidden_dim]

    def clear(self):
        self.activations.clear()

class SteeringHook:
    """Causal intervention hook that adds or subtracts a steering vector during forward pass."""
    def __init__(self, steering_vectors: Dict[int, torch.Tensor], multiplier: float = 1.0):
        self.steering_vectors = steering_vectors # layer_idx -> [hidden_dim]
        self.multiplier = multiplier
        self.enabled = True

    def __call__(self, module: nn.Module, inputs: Tuple[Any], output: Any):
        if not self.enabled:
            return output
        
        # Handle tuple output vs tensor output from transformer layer
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = ()
            
        # Add steering vector across sequence positions
        # Find which layer this hook is attached to via module metadata or dict matching
        for layer_idx, vec in self.steering_vectors.items():
            if vec.device != hidden_states.device:
                vec = vec.to(hidden_states.device)
            # Add vector to the latest token or all sequence tokens
            hidden_states = hidden_states + (self.multiplier * vec.unsqueeze(0).unsqueeze(0))
            
        if rest:
            return (hidden_states,) + rest
        return hidden_states

class ReasoningInterpHarness:
    def __init__(
        self, 
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.model_name = model_name
        
        # Auto-detect best available device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Loading {self.model_name} on {self.device} with {torch_dtype}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        self.num_layers = self._get_num_layers()
        self.hidden_dim = self._get_hidden_dim()
        print(f"Model loaded successfully! Layers: {self.num_layers}, Hidden Dim: {self.hidden_dim}")
        
        self.cache = ActivationCache()
        self._hook_handles = []
        self._steering_handles = []

    def _get_layers_module(self) -> nn.ModuleList:
        """Returns the module list of transformer decoder layers."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        else:
            raise ValueError(f"Unknown architecture for model {self.model_name}")

    def _get_num_layers(self) -> int:
        return len(self._get_layers_module())

    def _get_hidden_dim(self) -> int:
        return self.model.config.hidden_size

    def register_recording_hooks(self, layer_indices: Optional[List[int]] = None):
        """Attaches hooks to record residual stream outputs from specified layers."""
        self.remove_hooks()
        layers = self._get_layers_module()
        
        if layer_indices is None:
            # Default to tracking every 4th layer + final layer for memory efficiency
            layer_indices = list(range(0, self.num_layers, max(1, self.num_layers // 6)))
            if (self.num_layers - 1) not in layer_indices:
                layer_indices.append(self.num_layers - 1)
                
        for idx in layer_indices:
            layer = layers[idx]
            
            def make_hook(l_idx):
                def hook_fn(module, inp, out):
                    hidden = out[0] if isinstance(out, tuple) else out
                    if l_idx not in self.cache.activations:
                        self.cache.activations[l_idx] = []
                    # Detach and move to CPU to preserve GPU memory during generation
                    self.cache.activations[l_idx].append(hidden.detach().cpu())
                return hook_fn
                
            handle = layer.register_forward_hook(make_hook(idx))
            self._hook_handles.append(handle)

    def attach_steering_hook(self, layer_idx: int, steering_vector: torch.Tensor, multiplier: float = 1.0):
        """Attaches a causal steering hook to inject vector during generation."""
        layers = self._get_layers_module()
        target_layer = layers[layer_idx]
        
        hook = SteeringHook({layer_idx: steering_vector}, multiplier=multiplier)
        handle = target_layer.register_forward_hook(hook)
        self._steering_handles.append(handle)
        return hook

    def remove_hooks(self):
        """Removes all recording and steering hooks."""
        for handle in self._hook_handles:
            handle.remove()
        for handle in self._steering_handles:
            handle.remove()
        self._hook_handles.clear()
        self._steering_handles.clear()
        self.cache.clear()

    def generate_with_cache(
        self, 
        prompt: str, 
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        record_activations: bool = False
    ) -> Dict[str, Any]:
        """Generates reasoning response and optionally records residual stream activations."""
        self.cache.clear()
        if record_activations:
            self.register_recording_hooks()
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        raw_output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        generated_tokens = outputs[0][prompt_len:]
        completion_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Parse reasoning structure
        cot, final_answer = self.parse_reasoning_output(completion_text)
        
        result = {
            "prompt": prompt,
            "full_text": raw_output_text,
            "completion": completion_text,
            "cot": cot,
            "final_answer": final_answer,
            "token_count": len(generated_tokens),
            "activations": dict(self.cache.activations) if record_activations else None
        }
        
        if record_activations:
            self.remove_hooks()
            
        return result

    @staticmethod
    def parse_reasoning_output(text: str) -> Tuple[str, str]:
        """Extracts <think>...</think> block and the final extracted answer string."""
        think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if think_match:
            cot = think_match.group(1).strip()
            remaining = text[think_match.end():].strip()
        else:
            cot = ""
            remaining = text.strip()
            
        # Look for standard boxed answers or explicit 'The answer is X'
        boxed_match = re.findall(r"\\boxed\{(.*?)\}", remaining)
        if boxed_match:
            final_ans = boxed_match[-1].strip()
        else:
            # Fallback: extract last numeric token or sentence
            ans_match = re.findall(r"(?:answer is|equals|result is|=|is)\s*[:\$]?\s*([0-9\.\,\-]+)", remaining, flags=re.IGNORECASE)
            if ans_match:
                final_ans = ans_match[-1].strip().replace(",", "")
            else:
                # Last number in text
                numbers = re.findall(r"[-+]?\d*\.?\d+", remaining)
                final_ans = numbers[-1] if numbers else remaining[:50]
                
        return cot, final_ans
