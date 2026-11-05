# FastKernels: Benchmarking GPU Kernel Generation in Production

**Authors:** Gabriele Oliaro, Yichao Fu, May Jiang, Owen Lu, Junli Wang, Aurick Qiao, Zhihao Jia, Hao Zhang, Samyam Rajbhandari
**Venue:** NeurIPS 2026 Evaluations and Datasets Track (Withdrawn Submission), Submission #2832
**Forum:** https://openreview.net/forum?id=4FWYE316D7
**Keywords:** gpu kernels, cuda kernels, kernel benchmarking, CUDA generation, inference
**Primary Area:** SysML infrastructure
**Contribution Type:** Benchmark design and benchmark analysis

**TL;DR:** A GPU kernel benchmark grounded in production inference frameworks

## Abstract

LLM-based agents for GPU kernel generation are advancing rapidly, yet their progress is fundamentally constrained by the benchmarks they optimize against. Existing benchmarks are poorly aligned with production inference frameworks: they evaluate kernels on a single GPU with synthetic inputs, ignore the surrounding compilation stack, and reward replicating known optimizations rather than discovering new ones. The resulting reward signals are misleading; agents learn to generate kernels that score well in sandboxes but introduce interface incompatibilities, compilation-stack conflicts, and silent correctness degradation when integrated into real systems. We introduce FastKernels, a kernel benchmark built around a minimal set of 46 representative architectures spanning 8 categories, whose kernels collectively subsume those of 96.2% (409/425) of HuggingFace Transformers architectures. FastKernels doubles as a minimalistic, production-grade inference framework that runs at parity with hardened systems such as vLLM and SGLang on mainstream LLM serving and substantially exceeds upstream references on under-served architectures; each task's interface mirrors the corresponding module in the state-of-the-art library for its architecture family, enabling direct deployment of optimized kernels into production codebases. Evaluating state-of-the-art kernel agents on FastKernels, we find that even the strongest agent achieves only 0.94× aggregate speedup over production baselines, with weaker agents at 0.78× and 0.53×, confirming that benchmark-production misalignment is a critical bottleneck for the field. We release FastKernels as a stepping stone toward kernel agents whose benchmark gains translate directly into production throughput improvements. Code is available at https://anonymous.4open.science/r/fast-kernels.

## Scores at a glance

| Reviewer | Rating | Confidence |
|---|---|---|
| Sa5c | 2 (Reject) | 4 |
| AFFX | 3 (Borderline reject) | 4 |
| PkYB | 2 (Reject) | 3 |

**Decision:** Reject (paper subsequently withdrawn by authors)

---

## Paper Decision (Program Chairs, 24 Sept 2026)

**Decision:** Reject

**Metareview:**

This paper addresses a timely and important challenge by proposing FastKernels, a production-aligned benchmark-as-framework that evaluates LLM-generated GPU kernels against real inference baselines rather than sandbox or eager references. Reviewers acknowledge the substantial engineering effort behind its design, particularly its production-matched interfaces, realistic input shapes, correctness calibration pipeline, and diagnostic value for kernel deployment. However, the submission suffers from fundamental gaps in evaluation scope and baseline stability that undermine its central claims. The headline conclusion that agents cannot match production performance is drawn from only three tested systems and omits current frontier agents that already exceed these baselines on newer hardware. Additionally, pinning to a single H100 commit and vLLM revision creates an immediate staleness problem, as production kernels evolve rapidly across backends and architectures without any mechanism for tracking drift. The paper also overstates its architectural scope by evaluating only L1/L2 tasks despite claiming a multi-level compositional hierarchy, and it ambiguously positions itself between a benchmark and a framework. These need to be strongly addressed and convincingly argued, supported, or expanded in a rebuttal/revision.

**Final Justification:**

No reviewer supports accepting this paper. The authors should consider new ways to frame their work and evaluate it. Is it a system, and does the starting kernel/baseline matter? Can hypothetical future architectures be evaluated and shown? In its current form, the article is not convincing to the readers and leaves too many unanswered questions.

---

## Global Response by Authors (to AC and all reviewers)

We thank the reviewers and the AC. The reviews converge on one substantive question — *does the "0.94× headline" survive stronger agents, newer baselines, and newer hardware?* — so we answer it directly, with new experiments, before the per-reviewer replies.

**§0. What we concede, and how it sharpens the contribution.** The reviewers are right that "even the strongest agent achieves only 0.94×" describes the three agents we tested on H100, not the field. We ran three additional, stronger systems and report them honestly: over the kernels each produces *valid*, Codex GPT-5.5 reaches 1.46×, Claude Opus 4.6 1.13× (both on B200), and AKO4X 1.18× (H100, L1+L2). Newer agents **do** beat production on the kernels they can produce. We were wrong to compress the result into a single speedup scalar, and we will revise the abstract, introduction, and headline accordingly.

The finding that survives — and that we will make the paper's thesis — is on the **validity/coverage axis, not the speed axis**: agents are fast *where they are valid*, but validity collapses exactly on the production-shaped tasks that matter for deployment.

| Agent (new) | L1 valid | L2 valid | L3+L4 valid | all-reduce | Speed over valid |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Codex GPT-5.5 (B200) | 38/48 | 35/40 | 3/18 | 3/15 (invalid) | 1.46× |
| Claude Opus 4.6 (B200) | 34/48 | 24/40 | 3/18 | 15/15 (valid) | 1.13× |
| AKO4X (H100, L1+L2) | 26/48 | 5/40 | — | 15/15 (valid) | 1.18× |

Even the strongest frontier agent covers 3/18 L3+L4 targets and cannot produce a correct tensor-parallel all-reduce (Codex 3/15 scenarios). This is the gap FastKernels is built to measure, and no operator-level or PyTorch-eager benchmark surfaces it. We will present the leaderboard as three separate axes (coverage, correctness, speed-over-valid) rather than a single blended number, and state the scope of every number (agent set, level, hardware) inline.

| Agent  | L3 valid | L4 valid | Speed over valid |
| :---- | :---- | :---- | :---- |
| Codex GPT-5.5 (H200) | 11/13 | 2/5 | 0.731x |
| Claude Opus 4.6 (H200) | 9/13 | 2/5 | 0.833x |

**§1. Agent scope, and why AKO4X's failures are real signal, not harness bugs.** We extended the candidate set with AKO4X (a recent FlashInfer-Bench-oriented agent), run as one agent session per target against the same production baselines. On L1+L2 it produced valid kernels for 31/88 targets at 1.18× geomean over valid; 57/88 never reached a valid comparison. Crucially, **this is not a benchmark bug**: every reference that AKO4X's single-file adapter failed to instantiate runs correctly in the native FastKernels harness — Codex obtains 88/88 valid on the identical targets, including `linear`, `layer_norm`, `moe_align`, `flashinfer_decode/prefill`, and `fused_experts`. AKO4X (like KernelBench-ecosystem agents) assumes each task is a self-contained file with a simple reference, and cannot reconstruct production-shaped module baselines that require config objects, KV-cache wrappers, prepared/packed weights, or a distributed context. That reconstruction failure **is** the production-interface gap our benchmark is designed to expose. AlphaEvolve and KernelEvolve have no public release; we will disclose this and stop implying a complete field survey.

**§2. Baseline versioning and drift.** FastKernels is versioned: each release pins exact revisions of every reference library (vLLM, vllm-omni, SGLang, FLA, diffusers, …) in a shipped manifest, and the pinned hardware is part of that manifest (a B200 or MI300X release carries its own baselines and its own drift measurements — directly addressing per-hardware backend dispatch). A daily CI job re-benchmarks only the baselines against any new reference revision and computes a per-family *drift coefficient*; users can convert a speedup measured against the FastKernels baseline into an expected speedup against the current upstream revision. When any family's coefficient exceeds 10%, a new pinned release is prepared and author-reviewed. To estimate the resulting cadence we measured end-to-end drift between vLLM v0.18.0 (pinned at submission, April 2026) and v0.25.1 (July 2026; v0.26.0 is now the latest) on 8×H200, over the 16 reference models both revisions serve, with Tier-3 workloads, datasets, and metrics unchanged:

| Family | n | Throughput | Latency |
| :---- | :---- | :---- | :---- |
| I. LLM, dense + MoE | 4 | 1.14× | 1.16× |
| II. Linear-attn / SSM / hybrid | 5 | 1.09× | 1.01× |
| III. Generative vision / video / audio | 4 | 0.97× | 1.02× |
| IV. Multimodal / VLM | 3 | 0.97× | 0.99× |
| **Overall** | **16** | **1.05×** | **1.04×** |

Per-model geomean over scenarios, equal-weighted within family; >1 means v0.25 is faster (diffusion/TTS rows are vllm-omni 0.18→0.24). One family crossed the 10% threshold over three months and a second came within a point, implying a re-release roughly quarterly — the same cadence needed to add newly-released architectures regardless of drift, so this is not extra burden relative to any benchmark that stays current. The drift is concentrated, not uniform, which is why the coefficient is per-family.

**§3. L1–L4 scope, and reconciling it with the automated pipeline.** The submission scoped the *headline* to L1+L2 because existing agents are built for self-contained or fused kernels (most do not even evaluate KernelBench L4). This was a presentation choice about which numbers to headline, **not** a limitation of the pipeline: the L3/L4 harness is fully automated, and for this rebuttal we ran all four agents over the complete 106-target graph (48 L1 + 40 L2 + 13 L3 + 5 L4; 1,181 scenarios) end-to-end with no human review. We will fix the Limitations wording that implied L3/L4 needs human review (it does not; Sa5c is right that this conflicted with Contribution 4). The L3/L4 results are the point: validity collapses to 2/18, 0/18, 3/18, 3/18 for Dr. Kernel, KernelAgent, Codex GPT-5.5, and Claude Opus 4.6. L3/L4 are a next-generation *compositional-agent* stress test, and we now have the automated evidence to say so.

**§4. Positioning.** FastKernels is primarily a benchmark. That it can also run as a minimal end-to-end inference engine (in the spirit of mini-sglang) is a byproduct of the production-faithful design, not a competing framework contribution. We will restructure so the agent evaluation is the main result and the engine is described as the substrate that makes production-faithful measurement possible.

---

## Review: Reviewer Sa5c (27 Jun 2026)

**Rating:** 2: Reject
**Confidence:** 4

### Summary

The paper introduces FastKernels, a benchmark-as-framework for evaluating LLM-based GPU kernel generation agents against production baselines rather than PyTorch eager. It covers 46 model architectures across 8 categories with a compositional L1-L4 task hierarchy, and evaluates three kernel agents, finding that even the strongest achieves only 0.94x against production references on H100.

### Strengths

The paper identifies a real and underappreciated problem: existing kernel benchmarks reward sandbox performance that does not transfer to production. The engineering effort is substantial, covering 46 architectures with production-matched interfaces, multi-GPU communication kernels, and a thoughtful correctness calibration pipeline, and the L1-to-L2 regression analysis across agents provides actionable diagnostic value.

I appreciate the hardworking on the engineering side but the design itself (building a self-contained inference framework) comes with an inherent weakness of staleness. See more details in weakness.

### Weaknesses

**Major**

1. **The agent evaluation misses the strongest available agents, weakening the headline conclusion.** The paper evaluates three agents from the KernelBench ecosystem and concludes "even the strongest agent achieves only 0.94x," framing this as a statement about the field. But by submission time, agents from the FlashInfer-Bench ecosystem were already beating production FlashInfer expert kernels (AKO4X on 9 of 10 operator families on B200, MLSys 2026 contest agent-only winners). The 0.94x result characterizes the three agents tested, not the state of the field. FastKernels' broader architecture scope may still challenge these newer agents, but that hypothesis is untested.

2. **Production baselines create a staleness problem the paper does not address.** Production kernels are a moving target: vLLM's attention subsystem has 20 backend implementations and hundreds of commits in 2026. FastKernels pins to a single vLLM commit and a single GPU type (H100 SXM5), so its baseline begins drifting immediately. vLLM also dispatches different attention backends per hardware, so a kernel beating the H100 reference may face an entirely different production baseline on B200 or MI300X. No mechanism for tracking baseline drift is described. A benchmark whose selling point is production alignment but whose baselines go stale risks becoming the misalignment it set out to fix.

3. **The 0.94x result is partly a hardware-vintage artifact.** H100 has had three years of kernel tuning by NVIDIA, the vLLM/SGLang community, and FlashAttention maintainers, making its production baselines among the most optimized in existence. The 0.94x tells us "agents can't beat three years of hand-tuned H100 kernels," which is unsurprising and much less generalizable than the paper's framing. On newer hardware like B200, where baselines are less mature and new instruction paths remain underexploited, FlashInfer-Bench agents already beat production kernels. The paper's conclusion would look very different if evaluated on B200.

These three together tell one story: the paper's main empirical claim (agents can't beat production) is an artifact of which agents, which baselines (moving target), and which hardware were chosen (H100 only).

The Limitations section acknowledges the single-GPU-type and L1+L2 scope, but does not address how these choices affect the paper's headline conclusion.

**Minor**

4. **The compositional hierarchy (L1-L4) is claimed but only L1+L2 are evaluated.** The agent evaluation covers only L1+L2, so there is no evidence that compositional reuse across levels improves kernel quality. The paper attributes this gap to "compute and human-review time," but this conflicts with Contribution 4, which presents automated end-to-end validation as a solved problem. If L3/L4 still requires human review, the automated pipeline fails at the scope where it matters most; if it does not, the justification for skipping L3/L4 is weak.

5. **The "benchmark-as-framework" motivation overstates the gap with existing systems.** The paper implies vLLM and SGLang only serve LLMs, but vLLM already supports audio, vision, VLMs, retrieval, SSMs, and diffusion. The architectures vLLM genuinely cannot serve (robotics, 3D rendering, recommendation) are exactly where FastKernels reports its largest speedups, but those gains are over research repos, not hardened frameworks. For LLMs, where production alignment matters most, FastKernels sits at parity or behind (0.84x on DeepSeek-V3.2).

### Questions

See weakness.

### Other fields

- Limitations: yes
- Ethical Concerns: No or very minor ethics concerns only
- Reproducibility: Yes
- Dataset Assessment: Yes
- Paper Formatting Concerns: no

### Rebuttal by Authors (28 Jul 2026)

**W1: Frontier agents (FlashInfer-Bench ecosystem).** Thank you for raising the scope of the evaluated agent set. We added AKO4X (FlashInfer-Bench-oriented) on L1+L2 and Claude Opus 4.6 on the full 106-target graph on B200, and we re-ran Codex GPT-5.5 on B200 (Global §0–1). Over valid kernels they reach 1.18× (AKO4X), 1.46× (Codex), and 1.13× (Claude Opus 4.6). Coverage remains lower on L3+L4: Codex is valid on 3/18 targets and passes 3/15 tensor-parallel all-reduce scenarios, while Claude Opus 4.6 passes all 15 all-reduce scenarios. We will revise the headline to state the evaluated agents, level, and hardware explicitly, and report coverage, correctness, and speed as separate axes. AlphaEvolve and KernelEvolve are not open-sourced; we will disclose this exclusion.

**W2: Baseline staleness.** Please see Global §2 for the versioning, per-family drift coefficient, author-reviewed re-release mechanism, and per-hardware manifest that addresses vLLM's per-hardware backend dispatch. The measured v0.18.0→v0.25.1 drift is 1.05× overall, with the LLM family crossing the threshold. This suggests a quarterly cadence similar to the cadence needed to add new architectures.

**W3: Hardware vintage.** H100 baselines are among the most mature, so results against them characterize an established deployment setting. To examine hardware dependence, we generated fresh kernels on B200 against B200-compatible production references for all four agents (Global §0). On B200, the stronger agents exceed production on L1+L2 over valid kernels (Codex GPT-5.5: 1.37×/1.54× on L1/L2; Claude Opus 4.6: 1.16×/1.12×). Coverage is lower at L3/L4, with 3/18 valid targets for Codex. The all-reduce results also differ across agents: Codex passes 3/15 scenarios, KernelAgent and Dr. Kernel pass 0/15, and Claude Opus 4.6 passes 15/15. These results suggest that baseline maturity affects speed measurements, while coverage and correctness remain separate evaluation dimensions. We will present the submission headline as specific to its hardware and baseline maturity and report per-hardware results separately.

**W4: L3/L4 scope.** See Global §3. The automated pipeline ran all 106 targets for four agents on B200 with no human review; validity drops from L1→L2→L3/L4 for every agent (e.g., Codex GPT-5.5 38/48 → 35/40 → 3/18). We will correct the Limitations text that implied L3/L4 requires human review.

**W5: vLLM modality coverage.** vLLM serves modalities beyond LLMs and is already our reference for the non-LLM modalities it supports (Table 7: Mamba/Mamba2 SSMs, Whisper audio, and Qwen3-VL VLMs; vllm-omni for FLUX.1-Dev). We will revise the phrasing that implied otherwise. Gains above 1× occur where the strongest available reference is a research repository or non-serving library (Pi0, ColBERTv2, and 3DGS). Parity on mainstream LLM serving is a requirement for a valid harness rather than a claimed improvement. The 0.84× on DeepSeek-V3.2 is our engine's end-to-end serving path against a vLLM FP8 path and does not enter agent scoring; we include it to report performance where vLLM is the stronger reference.

---

## Review: Reviewer AFFX (25 Jun 2026)

**Rating:** 3: Borderline reject
**Confidence:** 4

### Summary

The paper argues that evaluating LLM-generated GPU kernels only as isolated operators is not sufficient for understanding whether those kernels are useful in production inference systems. Existing kernel benchmarks often use simplified interfaces, synthetic inputs, single-GPU settings, and weak baselines such as PyTorch eager. FAST KERNELS instead evaluates kernels in a more production-like setting: real model architectures, production-shaped module interfaces, realistic input shapes or captured tensors, stronger production baselines, and in some cases multi-GPU communication.

The motivation is a kernel can look fast in a standalone benchmark but fail to improve, or even harm, end-to-end inference once it interacts with other . Therefore, kernel-generation benchmarks should measure not only local operator speed, but whether generated kernels remain correct, compatible, and beneficial inside realistic inference pipelines.

### Strengths

- Claim that standalone kernel evaluation can be misleading. Pushes the field toward measuring deployable speedups rather than benchmark-only wins.
- FAST KERNELS makes the evaluation more realistic by deriving tasks from real architectures and by aligning interfaces with production modules from frameworks such as vLLM and SGLang.
- Benchmark with real model architectures, production-shaped module interfaces, realistic input shapes or captured tensors, stronger production baselines, and in some cases multi-GPU communication.

### Weaknesses

- This kind of benchmark is expensive and operationally complex. Running an end-to-end inference framework requires realistic serving infrastructure, careful environment control, and potentially single-node multi-GPU or even multi-node multi-GPU setups. It also depends heavily on the quality of the framework itself: if the framework is not truly production-grade, then the benchmark may still fail to reflect real deployment.
- Conversely, if a production framework is well designed with clean kernel abstraction boundaries, independent kernel-level evaluation may already be fairly predictive, and the added end-to-end overhead may not always be necessary.
- I would also like to see more direct evidence that FAST KERNELS catches failures missed by operator-only evaluation: for the same operators, how often does standalone evaluation look fine but end-to-end performance drops? What are the causes: LLM-generated kernel quality, flawed benchmark assumptions, interface mismatch, communication effects, or framework overhead? More systematic case studies would make the argument stronger.

### Questions

Check above.

### Final Justification

I will keep my score.

### Other fields

- Limitations: Check above.
- Ethical Concerns: No or very minor ethics concerns only
- Reproducibility: Yes
- Dataset Assessment: NA - no dataset included
- Paper Formatting Concerns: N/A

### Rebuttal by Authors (28 Jul 2026)

**Operational cost.** The three-tier design bounds routine cost: Tier 1 runs isolated kernels on production-derived shapes and captured tensors, so day-to-day iteration needs no serving infrastructure; end-to-end runs (Tiers 2–3) are for validation and final scoring only.

**Is operator-level evaluation already predictive with clean abstractions?** Our results show differences between isolated and end-to-end evaluation even on stacks with clean kernel boundaries. We quantify the requested cases by substituting five Tier-1-correct Codex L1 kernels individually into end-to-end Llama-3.1-8B serving:

| Kernel | Tier-1 | E2E throughput | E2E token alignment |
|---|---|---|---|
| rms_norm | correct | 0.82× | 1/3 scenarios |
| silu_and_mul | correct | 0.92× | 1/3 scenarios |
| rotary_emb | correct | 0.88× | 0/3 scenarios |
| embedding | correct | ~1.00× | aligned |
| store_kvcache | correct | ~1.00× | aligned |

Three of five pass isolated correctness but do not preserve end-to-end token alignment and have lower throughput; the other two are approximately neutral. Separately, KernelAgent's all-reduce passes its staged single-process checker but passes 0/15 scenarios in the 4-rank distributed harness. Repeating the check with the rebuttal's AKO4X candidates, the full staged L1–L4 set does not run correctly inside Llama-3.1 because an isolated-valid L3 decoder does not satisfy the production context contract. An L1-only set without rms_norm is E2E-correct in eager mode with 0.18× throughput, while an embedding-only subset runs under CUDA graphs with 1.007× throughput and 3/4 exact token matches on a 4-request smoke. On reduced E2E smokes, GLA-2.7B and YOLOv10n remain correct with throughput of 0.97× and 0.97×. The observed causes include production-contract violations, communication effects, data-dependent input mismatch, and launch or graph-capture interactions. We will add this as a systematic case study addressing how often isolated correctness differs from end-to-end behavior and why.

**End-to-end framework comparison.** Section 6.1 reports performance relative to vLLM/SGLang on mainstream LLM serving (Llama 1.04×, GPT-OSS 1.02×, Mixtral 0.97×) and cross-modality alignment results (Tables 7–8). These measurements provide context for interpreting the end-to-end substitution experiments.

---

## Review: Reviewer PkYB (14 Jun 2026)

**Rating:** 2: Reject
**Confidence:** 3

### Summary

FASTKERNELS is a GPU kernel benchmark that doubles as a minimalistic production-grade inference framework. It derives tasks top-down from 46 real model architectures across 8 categories, organized into a four-level compositional hierarchy (primitives, fused operators, layers, full models), and evaluates kernels against production baselines (vLLM, SGLang, etc.) rather than PyTorch eager or theoretical speed-of-light bounds. Running three existing kernel-generation agents on the benchmark, the authors show that even the strongest agent (Codex) achieves only 0.94x aggregate speedup, exposing a systematic gap between prior benchmark scores and production reality.

### Strengths

The core critique of existing benchmarks is compelling and well-motivated: rewarding sandbox performance rather than production deployability is a genuine problem, and the benchmark-as-framework design directly addresses it. Module-level compatibility with vLLM and SGLang enables direct kernel deployment without interface refactoring. The four-level compositional hierarchy supports kernel reuse across abstraction levels, a property missing in flat benchmarks. The MoE routing study in Appendix D makes a concrete case for why captured real inputs matter over synthetic alternatives. MACROEVAL's design separating correctness, coverage, and speedup with macro-family averaging and geometric means is methodologically sound, and the sensitivity analyses including threshold scaling, harness-gap accounting, and bootstrap CIs are thorough.

### Weaknesses

1. The paper's positioning is somewhat ambiguous: it presents as a benchmark contribution but substantial portions read as an inference framework paper, and the agent evaluation feels secondary to the framework demonstration.
2. The compositional hierarchy is the central methodological claim, yet L3 and L4 agent evaluations are absent. Without these results, the reuse argument currently lacks experimental support.
3. Evaluation conditions across agents are not fully controlled. Codex appears to use a multi-turn NCU-feedback loop while the other agents' iteration budgets are unspecified, and it is unclear how much of Dr. Kernel's 50/88 preflight rejections reflect harness interface issues rather than kernel quality.
4. The headline speedup numbers are somewhat difficult to interpret due to baseline heterogeneity. Mixing torch-eager references with highly optimized vendor kernels in a single geomean obscures where agents succeed or fail.
5. The paper's central thesis would benefit from stronger direct experimental support. Showing the same agents on both FASTKERNELS and a prior benchmark under identical conditions would make the misalignment argument considerably more convincing than citing previously reported numbers across different papers.
6. Three agent candidates with no human expert baseline is relatively limited for a benchmark paper. The absence of several frontier systems cited in related work such as AlphaEvolve and KernelEvolve is not explained.
7. The 96.2% HuggingFace coverage figure is acknowledged in the appendix as a partially spot-checked upper bound, but this qualification does not come through clearly in the abstract and introduction.

### Questions

1. Is there any evidence that evaluated agents exploited the compositional reuse path across levels? If not, it would be helpful to clarify whether the benchmark tooling actively supports such reuse or merely permits it structurally.
2. Could the authors provide a breakdown of Dr. Kernel's preflight rejections by failure reason? Understanding how many reflect harness interface issues versus genuine kernel invalidity would help interpret the coverage numbers more fairly.
3. Why were AlphaEvolve, KernelEvolve, and ASTRA excluded from evaluation? If there are practical barriers such as access or cost, disclosing these would help contextualize the candidate set.
4. The all-reduce case revealed a kernel that passed the staged checker but failed in the distributed harness. Has analogous silent correctness failure been checked for other distributed operators, and could the authors report results of any such systematic check?
5. The threshold sensitivity sweep is only feasible for Codex due to the binary checker design used for the other agents. Could the authors report what range of threshold choices would change Dr. Kernel or KernelAgent's correct kernel counts, even as a rough bound?
6. Could the authors report agent rankings under lambda=0 and lambda=1 separately in addition to the default blend? This would help assess whether the lambda=0.5 choice materially affects the conclusions.
7. The Scoredefault scalar combines speedup, correctness, and coverage multiplicatively. Could the authors report the three component metrics separately per agent in a more prominent table, so readers can assess tradeoffs that the scalar obscures?

### Other fields

- Limitations: yes
- Ethical Concerns: No or very minor ethics concerns only
- Reproducibility: Yes
- Dataset Assessment: NA - no dataset included
- Paper Formatting Concerns: No paper formatting concerns.

### Rebuttal by Authors (28 Jul 2026)

**W1: Positioning.** See Global §4; we will restructure so the agent evaluation is the main result and the engine is the measurement substrate.

**W2 / Q1: L3/L4 and compositional reuse.** The tooling supports reuse directly: the task graph is shared across levels and lower-level kernels can be imported into higher-level tasks, and we have evidence agents used it. In the L1+L2 runs, Codex imported its own L1 kernels in 34 of 40 L2 submissions (e.g., its fused_experts composes its L1 moe_align, moe_grouped_gemm, moe_sum, silu_and_mul; its attention block composes its flash-attention, KV-cache-store, and rms_norm kernels), while KernelAgent and Dr. Kernel never reused L1 kernels (their L2 imports only torch/triton). On the automated 106-target B200 run, validity decreases beyond L2 for every agent (Global §0/§3). We therefore present L3/L4 as a compositional-agent stress test rather than include it in the current-agent headline.

**W3 / Q2: Evaluation conditions and preflight rejections.** The multi-turn NCU-feedback loop (Appendix E) is our internal reference-population agent, not an evaluated agent; the three evaluated agents ran default published configurations. Budgets: Dr. Kernel used 1 sample and up to 3 generate-execute-feedback turns; KernelAgent used 1 generation round and 4 parallel workers; Codex used one session per target, the default configuration, and no execution feedback. The B200 protocols are documented identically in the campaign report. Dr. Kernel's 50 preflight rejections occur before kernel generation because the reference module does not load in its single-file sandbox: 41 constructor or instantiation interface mismatches, 8 targets requiring production runtime state (KV-cache wrappers, fused-MoE workspaces, or packed FP8/MXFP4 weights), and 1 distributed-only operation. Patching the constructor path makes the 41 interface-blocked targets available, after which Dr. Kernel obtains 0/41 correct and 10/41 compiling submissions. Among the original 38 attempted targets, 30/38 are incorrect: 17 have no runnable kernel within budget and 13 compile but are incorrect. Table 5 reports the rankings under alternative accounting policies for blocked targets.

**W4: Baseline heterogeneity.** We will add a table stratifying each agent's geomean by reference type. For Codex: 1.16× on torch-eager-wrapped families (n=22) vs 0.93× on vendor/hand-written kernels (n=26). For the other agents, over their correct L1 kernels:

| Agent | torch-eager refs | vendor / hand-written refs |
|---|---|---|
| KernelAgent | 0.76× (n=18) | 0.87× (n=8) |
| Dr. Kernel | 0.48× (n=7; 0.90× excl. the log_sigmoid 0.012× outlier) | 0.92× (n=1) |

For these two agents the strata are dominated by kernels below 1× rather than by reference type. Among their kernels above 1×, 7 of KernelAgent's 8 and Dr. Kernel's single kernel (l2_norm) use torch-eager or FLA-recurrent references.

**W5: Same agents on a prior benchmark, identical conditions.** We ran the three agents on KernelBench Level 1 (100 targets) with the exact FastKernels configurations (Codex GPT-5.5: one session/target, read-only sandbox, no execution feedback; KernelAgent: 1 round, 4 workers; Dr. Kernel: 1 sample, ≤3 generate-execute-feedback turns) and scored the emitted kernels with the same S / C / Coverage / Score_default metrics as kb_nano. On KernelBench L1 the standard harness loads every reference, so there are no preflight rejections and C = Coverage. Side by side at the matched kb_nano correctness tolerance (fp32 atol=1e-5, rtol=1e-3):

| Agent | kb_nano, H100 (S / Cov / Score) | KernelBench L1, H200 (S / Cov / Score) |
|---|---|---|
| Codex | 0.943× / 1.000 / 0.943 | 0.967× / 0.62 / 0.372 |
| KernelAgent | 0.777× / 0.318 / 0.091 | 0.676× / 0.67 / 0.303 |
| Dr. Kernel | 0.527× / 0.091 / 0.010 | 0.559× / 0.31 / 0.054 |

The geomean speed-over-correct ordering (Codex > KernelAgent > Dr. Kernel) is the same on both benchmarks, with similar S magnitudes for each agent. All three agents' correct kernels are below 1× on L1 (S = 0.97/0.68/0.56), as they are on kb_nano (0.94/0.78/0.53). Coverage differs more substantially: Codex's coverage decreases from 1.000 on kb_nano to 0.62 on L1, while KernelAgent is correct on 67 targets compared with Codex's 62. As a result, the Codex–KernelAgent Score ratio changes from approximately 10× on kb_nano to approximately 1.2× on L1 (0.372 vs 0.303). Codex retains the higher Score because its geomean speedup is 0.967× compared with KernelAgent's 0.676×. Dr. Kernel has the lowest Score on both benchmarks. At KernelBench's default fp32 tolerance (atol=rtol=1e-2), correct counts rise to 85 / 97 / 33 of 100 and S changes to 0.617× / 0.354× / 0.506×. The S ordering, Codex's Score lead (0.446 / 0.333), and Dr. Kernel's position (0.055) remain unchanged.

Conditions and caveats (for reproducibility). L1 ran on H200 vs the paper's H100 kb_nano; speedups are same-GPU custom-vs-torch-eager ratios, so the ranking is hardware-robust but absolute S is not cross-GPU comparable. KernelAgent was routed to Claude Opus 4.6 (the available backend) rather than the paper's 4.5; Codex is GPT-5.5 via codex-cli 0.145.0 at xhigh reasoning; Dr. Kernel is the released drkernel-14b checkpoint driven by a faithful reimplementation of its 3-turn generate-execute-feedback rollout (temp 1.0, top-p 0.95, verbatim prompt/feedback templates). Final speedups come from a clean, dedicated-GPU 100-trial timing pass, identical across agents. KernelAgent emitted 99/100 candidates (91_cumsum_reverse produced no parseable kernel across all seeds). Two Codex kernels flagged by KernelBench's own excessive-speedup guard are excluded as artifacts: 31_ELU, where a torch.rand input makes ELU an identity and yields a 962× no-op, and 40_LayerNorm.

**W6 / Q3: Candidate set and exclusions.** AlphaEvolve and KernelEvolve are not open-sourced (we will disclose this). AKO4X adds a fourth, more recent system: 31/88 valid on L1+L2 at 1.18× over valid. Eight of those 31 are unmodified-reference resubmissions, leaving 23/88 custom kernels (see Global §1). We plan a human-expert baseline on a subset of families for the revision.

**W7: Coverage figure.** We will carry the "spot-checked upper bound" qualification into the abstract and introduction.

**Q4: Other distributed operators.** All-reduce is the distributed operator currently implemented as an agent task. Its staged-versus-distributed results are KernelAgent/Dr. Kernel 0/15, Codex 3/15, and Claude 15/15 on the 4-rank harness, despite staged-checker passes. Extending the same check to reduce-scatter and all-to-all dispatch/combine is future work for the camera-ready.

**Q5: Threshold bounds for the binary-checker agents.** The staged checkers used by Dr. Kernel and KernelAgent emit binary pass/fail, so tightening tolerance can only reduce their correct counts; the reported 0.777× and 0.527× are therefore upper bounds under stricter tolerances. Per-scenario error magnitudes were not retained by these checkers, so the sweep cannot be recomputed post hoc; we will add per-scenario logging so the revision reports the full sweep for all agents (the Codex sweep in Appendix C already shows the ranking is stable from 0.25× to 5× tolerance).

**Q6: Rankings at λ=0 and λ=1.** The agent evaluation is kernel-level: each captured scenario contributes one measured timing ratio, so throughput and latency speedups coincide and the ranking is identical at λ=0, 0.5, 1. λ differentiates only Tier-3 end-to-end submissions, where throughput and latency are measured separately; we will state this in Section 6.2.

**Q7: Component metrics.** We will promote the per-agent component table to the main text and present Score_default as a secondary summary (kernel-level analogues on the 88-target graph: S = geomean over correct kernels, C = correct/attempted, Coverage = correct/88):

| Agent | S | C | Coverage | Score_default |
|---|---|---|---|---|
| Codex | 0.943× | 1.000 | 1.000 | 0.943 |
| KernelAgent | 0.777× | 0.368 | 0.318 | 0.091 |
| Dr. Kernel | 0.527× | 0.211 | 0.091 | 0.010 |

---

## Withdrawal by Authors (24 Sept 2026)

**Withdrawal Confirmation:** I have read and agree with the venue's withdrawal policy on behalf of myself and my co-authors.
