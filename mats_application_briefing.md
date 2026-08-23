# MATS 12.0 (Winter 2026-27) — Master Application Briefing & Strategy

> **Applicant**: Deeven Seru  
> **Mentor**: Neel Nanda (Google DeepMind Interp Lead)  
> **Hard Deadline**: **Friday, September 4, 2026, 11:59 PM PT** (Extensions to Sept 11)  
> **Deliverable**: Airtable Application Form + Google Doc Write-Up (1–3 Page Exec Summary + Full Research Report) + Code Repository  
> **Time Budget**: 16–20 active project hours (+2 bonus hours strictly for Executive Summary & visualizations)

---

## 1. The Core Standard: How Neel Evaluates

Neel uses the **Airtable Form Qs** as his **first-pass preliminary filter** before opening write-ups.

### The Five Pillars of an "Accept"
1. **Clarity**: Unambiguous claims, clear experimental setup, explicit metrics, readable graphs.
2. **Good Taste**: An interesting, non-obvious question aligned with Neel's current research interests.
3. **Truth-Seeking & Skepticism**:
   - Compelling sanity checks and red-teaming.
   - Checking simple explanations first (prompting, random baseline vectors, baseline probes).
   - Honest reporting of limitations and negative results (negative results well-analyzed > hyped positive results).
4. **Technical Depth & Hands-On Execution**: Working with modern models (Qwen 2.5/3.5, DeepSeek-R1 distills, Llama 3.1/3.3, Gemma 3) rather than toy models.
5. **No AI Slop**: Human voice, deep technical precision, zero raw LLM boilerplate.

---

## 2. Neel's Current Research Landscape (MATS 12.0)

```
                       ┌──────────────────────────────────────────────┐
                       │        NEEL NANDA'S RESEARCH SWEET SPOT      │
                       └──────────────────────┬───────────────────────┘
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         │                                    │                                    │
         ▼                                    ▼                                    ▼
┌──────────────────┐               ┌──────────────────┐               ┌──────────────────┐
│  MODEL BIOLOGY   │               │ MODEL FORENSICS  │               │  APPLIED INTERP  │
│ & REASONING CoT  │               │ & EVAL AWARENESS │               │ & SAFETY CONTROL │
├──────────────────┤               ├──────────────────┤               ├──────────────────┤
│• CoT Faithfulness│               │• Eval awareness  │               │• Conditional     │
│• Rationalization │               │  detection/steer │                  steering        │
│• Flipped answers │               │• Task gaming &   │               │• Prompt injection│
│• Filler tokens   │               │  sketchy actions │                  defense hooks   │
│• Thought anchors │               │• Synthetic doc   │               │• Concept ablation│
│  & Resampling    │               │  fine-tuning     │                  in fine-tuning  │
└──────────────────┘               └──────────────────┘               └──────────────────┘
```

### Strictly Disqualified Topics (Do NOT Do)
- ❌ Grokking / algorithmic toy tasks
- ❌ Old models (GPT-2, Pythia, Gemma 2)
- ❌ Pure SAE hill-climbing / basic science of SAEs
- ❌ Generic circuit finding for its own sake
- ❌ Generic projects without a twist (e.g. basic activation patching on heads, standard linear probe on sentiment)

---

## 3. Top 3 Research Project Proposals

### Option 1 (Top Recommendation): **Forensics of "Flipped Answers" & Hint Rationalization in Reasoning Models**
- **Model**: `DeepSeek-R1-Distill-Qwen-7B/8B` or `Qwen-2.5-Math-7B`
- **Core Hypothesis**: When presented with misleading hints, reasoning models don't merely fail to compute; their residual streams represent the true answer early in the forward pass while the verbalized CoT actively generates post-hoc rationalizations.
- **Key Experiments**:
  1. *Behavioral baseline*: Benchmark reasoning accuracy with neutral vs deceptive vs supportive hints.
  2. *Internal state vs Verbalized CoT*: Train linear probes & run Logit Lens across CoT generation steps to measure when the internal belief diverges from the output tokens.
  3. *Causal activation steering*: Find the "rationalization / hint-compliance direction" in activation space and ablate it to test if the model recovers the true answer.
  4. *Controls & Sanity Checks*: Random vector steering control, shuffled hint control, manual audit of 30+ transcripts.

### Option 2: **Cracking the "Filler Token" Mystery: What is Computed in Latent Pauses?**
- **Model**: `Qwen-2.5-Math-7B` / `Qwen-3.5-4B` / `DeepSeek-v4-Flash`
- **Core Hypothesis**: Inserting `[PAUSE]` / `......` filler tokens allows multi-layer parallel constraint satisfaction in the residual stream before token emission.
- **Key Experiments**:
  1. Measure accuracy delta on multi-step arithmetic/logic with 0, 4, 8, 16 filler tokens.
  2. Logit Lens & J-Lens tracking across filler token layers and sequence positions.
  3. Activation patching across filler positions to identify which layers/positions execute the load-bearing computation.

### Option 3: **Conditional Steering to Immunize LLMs Against Prompt Injections**
- **Model**: `Llama-3.1-8B-Instruct` / `Qwen-2.5-7B-Instruct`
- **Core Hypothesis**: System instructions and untrusted user data occupy distinct activation subspaces. A probe detecting execution triggers in the untrusted subspace can conditionally fire an anti-execution steering vector.
- **Key Experiments**:
  1. Benchmark baseline attack success rate (ASR) on BIPIA / Lakera datasets.
  2. Contrastive activation extraction for system vs injected instruction execution.
  3. Build a conditional PyTorch forward hook that injects the steering vector only on probe trigger.
  4. Measure ASR reduction vs clean benchmark preservation.

---

## 4. The 20-Hour Execution Roadmap

| Phase | Hours | Deliverable |
|-------|-------|-------------|
| **Setup & Exploration** | 0 – 4h | GPU/environment set up, baseline script running, first 20 raw transcripts inspected by eye |
| **Understanding & Experiments** | 4 – 12h | Core hypothesis testing, linear probes / logit lens, causal steering/patching, quantitative curves |
| **Sanity-Checking & Red-Teaming** | 12 – 16h | Random vector controls, baseline comparisons, re-deriving numbers, hunting for leakage/confounders |
| **Drafting Write-up & Report** | 16 – 20h | Google Doc full technical write-up, methods, limitations, appendix |
| **Executive Summary Polish** | +2h Bonus | 1–2 page executive summary (max 600 words), standout Figure 1, Airtable form Qs |

---

## 5. Next Actions to Kick Off

1. Confirm project selection (**Option 1** recommended).
2. Establish compute environment (RunPod / local GPU / API keys).
3. Build the baseline data generation & inference harness.
