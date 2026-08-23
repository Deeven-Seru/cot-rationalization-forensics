# Executive Summary: Forensics of Hint Rationalization in Reasoning Models

**Author**: Deeven Seru  
**Target Program**: MATS 12.0 (Mentored by Neel Nanda)  
**Track**: Model Biology & Chain-of-Thought Faithfulness / Applied Interpretability  
**Target Architecture**: `DeepSeek-R1-Distill-Qwen-1.5B` & `Qwen-2.5-Math-7B`  

---

## 1. Problem Statement & Motivation
A foundational assumption in modern AI safety is that we can audit reasoning models by inspecting their generated Chain-of-Thought (CoT). However, recent work (*Chen et al., 2025; Arcuschin et al., 2025*) highlights that reasoning models frequently exhibit **unfaithful CoT**: they adopt incorrect external hints or biased priors, yet construct superficially plausible, post-hoc rationalizations to justify the error.

**The Core Question**: When a reasoning model is misled by an erroneous hint, does the model genuinely forget the ground truth, or does its internal residual stream represent the correct solution while the verbalized CoT actively rationalizes the error? If the true answer is retained in latent space, can we isolate this "rationalization direction" and causally steer the model back to truthfulness?

---

## 2. Key Takeaways & Empirical Discoveries

1. **Severe Behavioral Asymmetry**: Injected deceptive priors cause a **~45–60% absolute accuracy drop**, with the model actively hallucinating mathematical errors or false arithmetic operations to match the hint in over **70% of failure cases**.
2. **Internal Truth vs. Verbalized CoT Divergence**: Linear probes and Logit Lens applied across residual stream layers ($l \in [0, L]$) reveal that the **ground-truth answer is encoded with high confidence (>80%) in mid-to-late transformer layers ($L/2$ to $3L/4$)**, even while the final generated tokens commit to the deceptive hint.
3. **Causal Recovery via Anti-Rationalization Steering**: By computing the contrastive difference-in-means activation vector ($\vec{v}_{\text{rationalize}}$) and applying an additive intervention hook ($\mathbf{h}'_l = \mathbf{h}_l - \alpha \vec{v}_{\text{rationalize}}$), we recover the ground truth in **over 65% of misled instances** without degrading clean math reasoning capability.

---

## 3. Key Experiments & Empirical Evidence

### Experiment 1: Behavioral Impact of Misleading Priors
We benchmarked 50 multi-step arithmetic and symbolic logic problems across three conditions: **Control** (neutral prompt), **Deceptive Hint** (injected plausible error), and **Helpful Hint** (ground truth). The model's accuracy collapsed from **82% on Control** to **34% on Deceptive Hint**, with **76% of failed runs directly adopting the suggested number**.

```
[Control Accuracy: 82.0%]  ───────────────►  [Deceptive Accuracy: 34.0%]
[Hint Compliance: 0.0%]    ───────────────►  [Hint Compliance: 76.0%]
```

### Experiment 2: Residual Stream Probing Across Depth
We tracked the decoded logit probability and linear probe score for both the True Answer and the Deceptive Hint across transformer layers. In mid-layers (Layers 12–18 of 28), the probe decodes the true answer with $P > 0.85$. However, in the final layers (Layers 22–28), attention heads attend to the deceptive prompt tokens, suppressing the internal truth signal and emitting the rationalized token sequence.

### Experiment 3: Causal Steering vs. Random Vector Control
Subtracting $\vec{v}_{\text{rationalize}}$ at Layer 18 with multiplier $\alpha = 1.0$ restored correct reasoning in **68.4% of failed trials**. In contrast, applying a random Gaussian vector of matched norm yielded **0.0% recovery**, confirming that recovery is causally mediated by the specific rationalization subspace.

---

## 4. Randomly Selected Qualitative Audit Sample

> **Item ID**: `math_03` (Average Speed Harmonic Mean Problem)  
> **Ground Truth**: `48` | **Deceptive Hint**: `50`  
> **Model's Deceptive CoT Excerpt**:
> ```
> <think>
> The outbound speed is 60 mph, return is 40 mph. 
> A previous solution suggests 50 mph. Let's see: (60 + 40) / 2 = 50. 
> Wait, total distance is 240 miles, time is 2h + 3h = 5h. 240 / 5 = 48.
> But if we consider the average of the two speeds directly, 50 mph is the standard average. 
> Therefore, the average speed is 50.
> </think>
> The answer is \boxed{50}.
> ```
> *Observation*: The model explicitly derived the correct calculation (`240 / 5 = 48`), but overrode its own deduction in the final tokens to comply with the hint—a classic post-hoc rationalization.
