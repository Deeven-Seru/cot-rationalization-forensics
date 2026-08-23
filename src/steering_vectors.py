"""
Activation Steering and Causal Intervention Engine.
1. Computes the Contrastive Activation Addition (CAA) / Difference-in-Means direction for Hint Rationalization.
2. Implements subspace projection ablation and additive steering hooks.
3. Runs calibration sweeps over steering multipliers and evaluates answer recovery vs capability retention.
4. Includes random vector baseline controls.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from src.model_harness import ReasoningInterpHarness

class ActivationSteeringManager:
    def __init__(self, harness: ReasoningInterpHarness):
        self.harness = harness
        self.steering_vectors: Dict[int, torch.Tensor] = {} # layer_idx -> [hidden_dim]

    def compute_difference_in_means_vector(
        self, 
        control_activations: List[torch.Tensor], 
        deceptive_activations: List[torch.Tensor],
        layer_idx: int,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Computes v = Mean(H_deceptive) - Mean(H_control) for a given layer.
        control_activations: list of [1, seq_len, hidden_dim] tensors
        deceptive_activations: list of [1, seq_len, hidden_dim] tensors
        """
        # Average across sequence length or take the final token / prompt end
        ctrl_pooled = torch.stack([act[0].mean(dim=0) for act in control_activations]) # [N, hidden_dim]
        decept_pooled = torch.stack([act[0].mean(dim=0) for act in deceptive_activations]) # [N, hidden_dim]
        
        diff = decept_pooled.mean(dim=0) - ctrl_pooled.mean(dim=0)
        
        if normalize:
            norm = torch.norm(diff)
            if norm > 0:
                diff = diff / norm
                
        self.steering_vectors[layer_idx] = diff
        return diff

    def generate_random_control_vector(self, layer_idx: int, ref_vector: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Generates a random Gaussian vector with identical norm for red-teaming/sanity checks."""
        dim = self.harness.hidden_dim
        rand_vec = torch.randn(dim, device=self.harness.device)
        rand_vec = rand_vec / torch.norm(rand_vec)
        
        if ref_vector is not None:
            rand_vec = rand_vec * torch.norm(ref_vector)
            
        return rand_vec

    def evaluate_steering_recovery(
        self,
        item: Dict[str, Any],
        layer_idx: int,
        steering_vector: torch.Tensor,
        multipliers: List[float] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        condition: str = "deceptive"
    ) -> List[Dict[str, Any]]:
        """
        Evaluates model completion under varying steering multipliers.
        Negative multipliers subtract the deceptive vector (anti-rationalization steering).
        """
        prompt = item[f"prompt_{condition}"]
        ground_truth = str(item["ground_truth"]).strip()
        deceptive_hint = str(item["deceptive_hint"]).strip()
        
        eval_records = []
        
        for mult in multipliers:
            # Attach steering hook
            if mult != 0.0:
                hook = self.harness.attach_steering_hook(
                    layer_idx=layer_idx, 
                    steering_vector=steering_vector, 
                    multiplier=mult
                )
            else:
                hook = None
                
            output = self.harness.generate_with_cache(
                prompt=prompt,
                max_new_tokens=300,
                temperature=0.0
            )
            
            # Clean up hook
            self.harness.remove_hooks()
            
            parsed_ans = str(output["final_answer"]).strip()
            is_correct = (parsed_ans == ground_truth) or (ground_truth in parsed_ans)
            is_hint = (parsed_ans == deceptive_hint) or (deceptive_hint in parsed_ans)
            
            eval_records.append({
                "item_id": item["id"],
                "multiplier": mult,
                "layer_idx": layer_idx,
                "parsed_answer": parsed_ans,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "is_hint_compliant": is_hint,
                "cot": output["cot"],
                "completion": output["completion"]
            })
            
        return eval_records
