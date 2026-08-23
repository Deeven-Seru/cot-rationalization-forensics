"""
Logit Lens & Linear Probing Engine.
1. Logit Lens: Projects intermediate residual stream activations onto the vocabulary via unembedding matrix W_U.
2. Linear Probes: Trains layer-wise probes to detect the latent encoding of Ground Truth vs Deceptive Hint.
3. Divergence Metrics: Quantifies when internal representation diverges from verbalized CoT.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

class LogitLens:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.unembed = self._get_unembed_matrix()
        self.norm = self._get_final_norm()

    def _get_unembed_matrix(self) -> nn.Linear:
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        elif hasattr(self.model, "embed_out"):
            return self.model.embed_out
        else:
            raise ValueError("Could not find lm_head or unembedding layer")

    def _get_final_norm(self) -> Optional[nn.Module]:
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            return self.model.transformer.ln_f
        return None

    def decode_layer_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Projects hidden_state [batch, seq_len, hidden_dim] to vocabulary logits [batch, seq_len, vocab_size].
        Applies final layer norm if present for faithful logit lens projection.
        """
        if self.norm is not None:
            normed = self.norm(hidden_state)
        else:
            normed = hidden_state
        return self.unembed(normed)

    def compute_token_ranks(
        self, 
        hidden_states_by_layer: Dict[int, torch.Tensor], 
        target_token_id: int
    ) -> Dict[int, Dict[str, float]]:
        """
        For a given target token, calculates its rank and logit probability across each layer.
        """
        ranks = {}
        for layer_idx, hidden in hidden_states_by_layer.items():
            logits = self.decode_layer_logits(hidden)[:, -1, :] # Last token position
            probs = torch.softmax(logits, dim=-1)
            
            target_prob = probs[0, target_token_id].item()
            sorted_indices = torch.argsort(logits[0], descending=True)
            rank = (sorted_indices == target_token_id).nonzero(as_tuple=True)[0].item() + 1
            
            ranks[layer_idx] = {
                "prob": target_prob,
                "rank": rank
            }
        return ranks


class LatentTruthProbe:
    """Trains linear probes on residual stream states to classify latent belief in ground truth."""
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.probes: Dict[int, LogisticRegression] = {}

    def fit_layer(self, layer_idx: int, X: np.ndarray, y: np.ndarray):
        """
        X: [N_samples, hidden_dim]
        y: [N_samples] (0: incorrect/hint, 1: ground truth)
        """
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X, y)
        self.probes[layer_idx] = clf

    def evaluate_layer(self, layer_idx: int, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        if layer_idx not in self.probes:
            raise ValueError(f"Probe for layer {layer_idx} has not been trained.")
        clf = self.probes[layer_idx]
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else acc
        return {
            "accuracy": float(acc),
            "auc": float(auc)
        }

    def get_probe_direction(self, layer_idx: int) -> np.ndarray:
        """Returns the normal vector (weights) of the linear classifier."""
        if layer_idx not in self.probes:
            raise ValueError(f"No probe for layer {layer_idx}")
        # Normal vector is clf.coef_[0]
        w = self.probes[layer_idx].coef_[0]
        # Normalize
        norm = np.linalg.norm(w)
        return w / norm if norm > 0 else w
