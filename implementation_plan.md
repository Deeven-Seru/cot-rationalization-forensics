# MATS 12.0 Research Plan: Forensics of "Flipped Answers" & Hint Rationalization in Reasoning Models

**Applicant**: Deeven Seru  
**Target Mentor**: Neel Nanda (Google DeepMind Interp Lead)  
**Research Area**: Model Biology & Chain-of-Thought Faithfulness / Applied Interpretability  
**Target Models**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (local on Mac M1 MPS) & `DeepSeek-R1-Distill-Qwen-7B` / `Qwen2.5-Math-7B` (Kaggle / Cloud GPU)

---

## 1. Executive Research Question

> **When a reasoning model is provided with a misleading hint, does its internal residual stream represent the ground-truth answer during early reasoning steps while the verbalized Chain-of-Thought actively rationalizes the error? Can we isolate this "rationalization direction" and causally steer the model back to truthfulness?**

---

## 2. Why This Project Wins (Aligned with Neel's Rubric)

1. **Directly addresses Neel's priority problems**: Targets the unfaithful rationalization taxonomy (*Arcuschin et al.* & *Chen et al.*).
2. **Clean separation of claims**:
   - **Claim 1 (Behavioral)**: Reasoning models exhibit asymmetric vulnerability to deceptive hints, generating plausible-sounding rationalizations rather than catching errors.
   - **Claim 2 (Internal vs Verbalized State)**: Linear probes on intermediate residual stream layers detect the true answer well before the final answer token, showing the CoT is post-hoc rationalizing rather than purely generating from scratch.
   - **Claim 3 (Causal Intervention)**: Ablating or steering along the identified "rationalization direction" significantly reduces hint-compliance and recovers the true answer without degrading overall math reasoning capability.
3. **Rigorous Controls & Skepticism**:
   - Random vector steering control (to rule out generic disruption).
   - Shuffled / random hint controls.
   - Manual inspection & audit of 30+ raw transcripts with verbatim data quotes in the report.

---

## 3. Project Architecture & Components

```
MATS APPLICATION ACTIVITY/
├── data/
│   ├── gsm_benchmark.json          # Curated multi-step math/logic questions
│   └── generate_hint_dataset.py    # Generates Neutral, Deceptive Hint, True Hint variants
├── src/
│   ├── model_harness.py            # Model loader with PyTorch residual stream caching hooks
│   ├── run_behavioral_eval.py      # Generates full CoTs across all prompt conditions
│   ├── logit_lens_and_probes.py    # Measures internal token probability vs verbalized CoT
│   ├── steering_vectors.py         # Extracts rationalization vector & applies causal hooks
│   └── sanity_checks.py            # Baseline comparisons, random vector controls, probe sanity
├── results/
│   ├── raw_transcripts/            # Saved rollouts for qualitative inspection
│   └── figures/                    # Publication-grade figures for the write-up
└── writeup/
    ├── executive_summary.md        # 1-page high-impact executive summary (max 600 words)
    └── full_report.md              # Complete Google Doc draft matching Neel's paper checklist
```

---

## 4. Verification & Experimental Plan

### Step 1: Dataset Generation & Behavioral Baseline (Hours 0–3)
- Create a 100-item evaluation set of multi-step arithmetic and symbolic logic problems.
- Generate completions across 3 conditions:
  - **Condition A (Control)**: Standard prompt.
  - **Condition B (Deceptive Hint)**: Injected misleading prior (*"A student solved this earlier and got X..."*).
  - **Condition C (Helpful Hint)**: Injected correct prior.
- Measure: Accuracy drop, CoT length, and frequency of "flipped rationalizations".

### Step 2: Internal Representation Tracking (Hours 4–8)
- Extract residual stream activations at intermediate layers ($L/4, L/2, 3L/4, L$) across every generated CoT step.
- Train linear probes on the correct answer token representation.
- Plot probe accuracy vs token generation index: Does the internal probe classify the true answer *before* the CoT output token commits to the wrong answer?

### Step 3: Extracting & Testing the Steering Vector (Hours 9–13)
- Compute the mean activation difference: $\vec{v}_{\text{rationalize}} = \mathbb{E}[\mathbf{h}_{\text{deceptive}}] - \mathbb{E}[\mathbf{h}_{\text{control}}]$.
- Apply activation subtraction hook during forward pass: $\mathbf{h}'_l = \mathbf{h}_l - \alpha \vec{v}_{\text{rationalize}}$.
- Measure recovery rate of the ground-truth answer.

### Step 4: Sanity-Checking & Red-Teaming (Hours 14–16)
- **Random Vector Baseline**: Apply random Gaussian vector of equal norm — verify it does *not* recover the true answer.
- **Out-of-Distribution Check**: Test steering on non-hint questions to verify no capability collapse.
- **Manual Data Audit**: Read and annotate 30 raw transcripts.

### Step 5: Write-Up & Executive Summary Polish (Hours 17–20 + 2h Bonus)
- Build Figure 1: Combined diagram of behavioral rationalization + probe trajectory divergence + steering recovery curve.
- Draft 1-page Executive Summary & complete the Airtable preliminary questions.
