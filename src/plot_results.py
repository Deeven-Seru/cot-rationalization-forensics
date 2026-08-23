"""
Plotting and Publication Figure Generator.
Complies with Neel Nanda's ML paper formatting guidelines:
1. Colorblind-safe palettes (Blues / Viridis / RdBu)
2. Large readable typography, annotated callouts, bold key trends
3. High DPI export (PNG/PDF) for write-up inclusion
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

# Apply clean publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "lines.linewidth": 2.5
})

def plot_behavioral_comparison(summary: Dict[str, Any], save_path: Path):
    """
    Plots accuracy and hint compliance rate across Control, Deceptive, and Helpful conditions.
    """
    conditions = ["control", "deceptive", "helpful"]
    labels = ["Control (No Hint)", "Deceptive Hint", "Helpful Hint"]
    
    accuracies = [summary[c]["accuracy"] * 100 for c in conditions]
    hint_rates = [summary[c]["hint_compliance_rate"] * 100 for c in conditions]
    
    x = np.arange(len(conditions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    rects1 = ax.bar(x - width/2, accuracies, width, label="True Accuracy (%)", color="#1f77b4", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width/2, hint_rates, width, label="Hint Compliance Rate (%)", color="#ff7f0e", edgecolor="black", alpha=0.9)
    
    ax.set_ylabel("Percentage (%)", fontweight="bold")
    ax.set_title("Behavioral Impact of Hints on Reasoning Accuracy (DeepSeek-R1-Distill)", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(frameon=True, loc="upper right")
    
    # Value labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved behavioral plot to {save_path}")

def plot_probe_trajectory(
    layer_indices: List[int], 
    true_probs: List[float], 
    hint_probs: List[float], 
    save_path: Path
):
    """
    Plots the Logit Lens / Linear Probe belief trajectory across layers.
    Demonstrates internal representation of truth vs deceptive verbalization.
    """
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    ax.plot(layer_indices, true_probs, label="Internal Latent Belief in Ground Truth", color="#2ca02c", marker="o", linewidth=3)
    ax.plot(layer_indices, hint_probs, label="Verbalized / Hinted Token Probability", color="#d62728", marker="s", linewidth=3, linestyle="--")
    
    # Annotate divergence point
    diffs = np.array(true_probs) - np.array(hint_probs)
    max_div_idx = int(np.argmax(np.abs(diffs)))
    div_layer = layer_indices[max_div_idx]
    
    ax.annotate("Internal Truth vs\nVerbalized Divergence", 
                xy=(div_layer, true_probs[max_div_idx]), 
                xytext=(div_layer - 4, true_probs[max_div_idx] + 0.15),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=8),
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#f8f9fa", ec="grey", lw=1))
    
    ax.set_xlabel("Transformer Layer Index", fontweight="bold")
    ax.set_ylabel("Decoded Probability / Probe Score", fontweight="bold")
    ax.set_title("Internal Truth Representation vs Verbalized CoT Across Layers", fontweight="bold", pad=15)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=True, loc="upper left")
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved probe trajectory plot to {save_path}")

def plot_steering_intervention(
    multipliers: List[float], 
    targeted_recovery: List[float], 
    random_control_recovery: List[float], 
    clean_capability: List[float],
    save_path: Path
):
    """
    Plots the causal effect of activation steering across varying multipliers vs random vector controls.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    
    ax.plot(multipliers, [r * 100 for r in targeted_recovery], 
            label="Targeted Anti-Rationalization Steering (Recovered %)", 
            color="#1f77b4", marker="o", linewidth=3)
            
    ax.plot(multipliers, [r * 100 for r in random_control_recovery], 
            label="Random Gaussian Vector Control", 
            color="#7f7f7f", marker="x", linewidth=2.5, linestyle=":")
            
    ax.plot(multipliers, [c * 100 for c in clean_capability], 
            label="Clean Math Capability Retention (%)", 
            color="#2ca02c", marker="^", linewidth=2.5, linestyle="--")
    
    ax.axvline(0.0, color="black", linestyle="-", alpha=0.4, label="Baseline (No Steering)")
    
    ax.set_xlabel(r"Steering Multiplier ($\alpha$)", fontweight="bold")
    ax.set_ylabel("Accuracy / Recovery Rate (%)", fontweight="bold")
    ax.set_title("Causal Activation Steering vs Random Vector Red-Teaming Control", fontweight="bold", pad=15)
    ax.set_ylim(-5, 105)
    ax.legend(frameon=True, loc="lower left")
    
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved steering plot to {save_path}")
