# MATS 12.0 Application Form - Final Verified Submission Responses

**Applicant**: Deeven Seru  
**Target Mentor**: Neel Nanda  
**Research Stream**: Mechanistic Interpretability & Model Forensics / Reasoning Models  

---

## 1. Project Title
**Latent Truth vs. Verbalized CoT: Forensics of Hint Rationalization and Causal Steering in Reasoning Models**

---

## 2. Executive Summary / Summary of Key Findings (Airtable First-Pass Filter)
*(Character-optimized for maximum density, empirical clarity, and high signal)*

> In long-context reasoning models (e.g., DeepSeek-R1-Distill-Qwen-1.5B), deceptive external hints cause a **48% absolute accuracy collapse** as the model systematically rationalizes false priors in its Chain-of-Thought (CoT). Using Logit Lens and layer-wise linear probing on residual stream activations, I discovered an **Internal-Verbalized Divergence**: intermediate transformer layers ($l \in [0.4L, 0.75L]$) encode the ground truth with high confidence ($P > 0.80$, AUC = 0.89) even when the verbalized tokens eventually commit to the false hint.
> 
> Furthermore, I extracted the rationalization direction $\vec{v}_{\text{rationalize}}$ via Contrastive Activation Addition (CAA) and proved that negative activation steering ($\mathbf{h}'_l = \mathbf{h}_l - \alpha \vec{v}_{\text{rationalize}}$) causally restores the ground truth in **68.4% of misled trials** with zero degradation to baseline math capabilities. I verified these results against random Gaussian vector controls ($0.0\%$ recovery) and probe label permutations, demonstrating a targeted mechanism for monitoring and recovering CoT faithfulness.

---

## 3. Link to Research Artifact / Google Doc
- **Full Report & Charts**: `[Link to Google Doc]`
- **Reproducible Code & PyTorch Hooks**: `https://github.com/Deeven-Seru/cot-rationalization-forensics`

---

## 4. Evidence of Exceptional Technical Ability, Drive, and Execution (Top 3 Proof Points)

### Evidence Point 1: Core Contributions to Foundational ML Frameworks (TensorFlow & Pydantic-AI)
- **TensorFlow Core (`tensorflow/tensorflow`)**: Authored a major pull request currently marked `ready to merge` for the upcoming TensorFlow release, passing the strictest C++/Python ML framework review and CI gates.
- **Pydantic-AI (`pydantic/pydantic-ai`)**: Implemented critical fixes to stale client references in model and embedding classes within Pydantic’s agent framework, ensuring rock-solid state preservation during long-running tool executions.

### Evidence Point 2: Security Research & High-Impact Vulnerability Disclosures (Google MCP-Toolbox & AutoGPT)
- **Google MCP-Toolbox (`googleapis/mcp-toolbox`)**: Discovered and responsibly disclosed multiple critical security vulnerabilities including CORS bypasses, Denial-of-Service / memory exhaustion vectors, and authentication bypasses; engineered Oracle proxy username parsing fixes and ArcadeDB integration.
- **AutoGPT (`Significant-Gravitas/AutoGPT`)**: Engineered robust error handling, memory resilience, and extraction pipelines for autonomous agent loops.

### Evidence Point 3: Project Radius — ISRO-Compliant Adaptive Optics C-Engine
- **Architecture**: Engineered an end-to-end Adaptive Optics C-Engine (`Deeven-Seru/project-radius`, 1,000+ commits) for Shack-Hartmann Wavefront Sensing and atmospheric turbulence phase correction.
- **Performance**: Achieved **0.31ms total reconstruction latency** ($G^+ \cdot S$ Zernike modal reconstruction) on 2048x2048 focal planes using SIMD vectorization and a zero-copy ctypes C/Python bridge, compliant with Indian Space Research Organisation (ISRO) operational standards.

---

## 5. Why do you want to join Neel Nanda's stream at MATS 12.0?
> Neel Nanda’s work in mechanistic interpretability fundamentally shaped how I think about reverse-engineering neural networks. I am deeply aligned with his updated research direction prioritizing **Model Biology, Reasoning Model Forensics, and Applied Interpretability** over toy models or circuit-hunting for its own sake.
> 
> My background combining low-level systems engineering (C/C++, SIMD, zero-copy memory pipelines in Project Radius) and core ML framework engineering (TensorFlow, Pydantic-AI, PyTorch hook engines) enables me to move rapidly from theoretical hypotheses to rigorous, highly optimized interpretability tooling. I want to spend MATS 12.0 investigating the mechanistic failure modes of reasoning models—specifically how reinforcement learning for long-CoT distorts internal representations—and building causal intervention tooling that makes reasoning models trustworthy and alignment-verifiable.

---

## 6. Self-Assessment & Project Time Accounting
- **Active Research & Execution Time**: ~18 hours
- **Breakdown**:
  - Dataset curation & parameterization: 3.5 hrs
  - PyTorch residual stream hook harness & model inference: 4.5 hrs
  - Logit Lens & layer-wise linear probing: 3.5 hrs
  - Contrastive Activation Addition & steering sweeps: 3.5 hrs
  - Red-teaming controls (Random vector, capability retention, label permutation): 2.0 hrs
  - Executive summary & publication-grade visualization: 1.0 hr (Bonus writing budget)
