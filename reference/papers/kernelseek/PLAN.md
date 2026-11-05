# Paper plan — "Ask, Don't Equip" (MLSys 2027)

## 0. Hard facts from the CFP (checked 2026-10-28)
- **Paper deadline: Fri Oct 30 2026, 20:00 UTC** (OpenReview, no separate abstract deadline). Rebuttal Jan 18–21 2027.
- 10 pages, two-column, excluding references; MLSys **2025** style files (what `main.tex` uses). Appendices unlimited and optional for reviewers. Check whether they go in the same PDF or are uploaded separately.
- Double-blind: anonymize the kernel/trace release (e.g. anonymous.4open.science). Don't self-cite in a way that identifies us; "our prior agent" is fine.
- **A mandatory "AI Usage Disclosure" section goes after the references** (`sections/ai-disclosure.tex`).
- Papers written mainly by generative models are prohibited. AI helped draft this skeleton and the intro, so the authors must rewrite/own the text and disclose this.
- Every bib entry needs the full author list (see `TODO-authors` entries in `references.bib`).

## 1. Thesis and story arc
**Agent name: KernelSeek** (`\sys` macro in `main.tex`; the tool-rich baseline is still `\baseline` = placeholder "Arsenal").

**Scope.** KernelSeek is a general agent for optimizing CUDA kernels: input is a working seed kernel, a scenario set and a goal; it needs no operator-specific knowledge and no faster reference implementation. FlashAttention-3/4 is the case study, chosen because an open upstream makes the result measurable. The upstream is a yardstick and correctness oracle for the evaluation, not an input to the method. Every section must keep this distinction.

**Thesis.** For deep system-level kernel optimization, the agent's bottleneck is *direction*, not *information*. Asking the right questions (Ask) beats equipping an autonomous agent with tools and reference material (Equip).

Arc:
1. Deep targets are different, and the field's answer is "Equip". FA3/FA4 is our instance of a deep target.
2. We built a serious Equip agent. It plateaued at 0.50× of FA3.
3. Agents fail in five recurring ways that more tools don't fix: wrong layer, false ceilings, premature closure, narrow benchmarks, reinventing instead of reading.
4. Ask fixes these. It is a minimal harness, a verifiable goal, and questions at decision points; it composes with AVO-style evolution.
5. Evidence:
   - controlled head-to-head (same evaluator, configs, machine and week);
   - the kernels (1.15× FA3, 1.46× FA4);
   - the mechanism (case studies plus statistics).
6. Release kernels and traces as seeds.

Title alternatives: "Ask, Don't Equip: …" (current) · "Interrogate, Don't Equip" · "Questions, Not Tools: Steering Coding Agents to the Fastest Open-Source FlashAttention".

## 2. Section plan and page budget (10 pages)
| § | File | Pages | Content |
|---|---|---|---|
| 1 | `01-introduction.tex` | 1.5 | **Written.** Tighten to about 1.25 pages once Fig. 1 exists. |
| 2 | `02-background.tex` | 1.0 | Why FA3/FA4 are hard; the Equip recipe; motivating example |
| 3 | `03-kernelseek.tex` | 2.0 | **Written.** KernelSeek (merged): Ask vs. Equip common frame; harness and Stop-hook goal; question bank (what a good question does + taxonomy); when to ask; skill; plus evolution; hypotheses H1–H5 |
| 4 | `04-baseline.tex` | 1.0 | **Written.** Baseline design (loop, levers, equipment table), outcome, why it stalled (five failure modes with numbers), lessons carried into KernelSeek |
| 5 | `05-kernels.tex` | 1.25 | **Written.** Case study: task, what the agent built, results vs upstream / fastest open-source / fastest available (Tab. 1 + Fig. `blackwell_speedups`), release. Details in Appendix A |
| 6 | `06-evaluation.tex` | 3.0 | **Written.** Setup; RQ1 pointer; RQ2 head-to-head (Tab. h2h, Fig. 1, Tab. tools); RQ3 anatomy (Fig. messages, Tab. cases); RQ4 evolution (Fig. evo); RQ5 transfer + planned automation ablation |
| 7 | `07-discussion.tex` | 0.5 | Why questions work; "a human did it" objection; when Equip wins; threats |
| 8 | `08-related.tex` | 0.75 | **Written.** Kernel agents along the structure--freedom axis; LLMs as optimizers; questions/critique/self-correction; scaffolding; attention kernels |
| 9 | `09-conclusion.tex` | 0.2 | |

Figures and tables (target 4 figures + 3 tables):
- **Fig 1 (teaser):** Ask vs Equip trajectories on the same 26-config evaluator over wall-clock hours, with questions marked.
- **Fig 2:** the two loops side by side (Equip: brain ← NCU parser, SASS diff, doc RAG, ref-lib hints, knowledge store, levers; Ask: agent ← eval command + question).
- **Fig 3:** `scripts/plot_casestudy.py` -> `figures/casestudy.pdf` (full width, 2x2): geomean per set vs upstream / fastest open-source / fastest available, and sorted per-scenario speedup vs fastest available, for Hopper (top) and Blackwell (bottom). Data in `paper/data/{hopper,blackwell}` (Hopper = per-scenario medians of `runs_h200/hopper-20261030T0535Z` rep-1..3, via `scripts/prepare_hopper.py`).
- **Fig 4:** question-category histogram, or a timeline of questions vs geomean for one campaign.
- **Tab 1:** question taxonomy. **Tab 2:** head-to-head. **Tab 3:** per-group perf summary.

## 3. Evidence inventory (all numbers to re-verify before submission)
**Equip baseline** (`agent/`, checkpoint `run_20260616_011948`; scripts in `analysis/baseline/`):
- claude-opus-4-8, effort xhigh for steps 0–6 and high afterwards; fresh session per step; 16 orchestrator levers.
- **The LLM was only active for steps 0–132.** Steps 133–312 are an OAuth-expired outage in which the loop re-scored old kernels. Report 133 steps / about 62 h / about $660, and shade or truncate the plot.
- Best geomean 0.502 and min 0.243 over 26 configs, with 0 configs ≥1.0×. The champion was selected by noise: step 112 is byte-identical to step 104, and 54 re-measurements give a mean of about 0.49.
- All progress happened by step 35. 312/313 candidates were correct. The 100-token decode gate was never reached.
- 37% of each prompt was pushed reference material. 82% of tool-result characters were re-reading its own kernel. The FA3 source was never opened.
- It declared "hardware walls". The memory store contained misleading lessons. 36 of 37 rewrite branches were seeded from the same step.
- 3 human rollbacks; 11 restarts.

**Ask traces** (`flash-cuda/reference/flash-cuda-chats/`; scripts and labeled turns in `analysis/traces/`):
- 16 sessions (9 Claude Code, 7 Cursor) from Jun 17 to Aug 7 2026; 157 human messages; 11,532 agent tool calls.
- Primary act of each human message: 25.5% questions, 36.9% directives; 37.6% are phrased as questions.
- **Head-to-head:** Cursor session 69fff5d7 used the baseline's own evaluator (`python -m fa3agent.trial`, `FA3_TRIAL_CONFIG_SET=full`). It went from 0.44 (Jun 21 03:29 UTC) to 0.96 (Jun 21 07:35), 0.97 (Jun 22) and 1.128 (Jun 23 about 03:00), with a worst case of about 0.87 (chunk16k). That took 58 human messages from Jun 18 00:36 to Jun 23 03:00 UTC.
- Case studies verified in the traces: coverage question (40/40 → 52/58 → 58/58); SOL-ExecBench real shapes; premature "goal met"; split-KV "ceiling" → host `at::full`; gap triage → scalar fallback; FA4 2-SM; tcgen05.wait deadlock; hd256 "floor" overturned.
- **Not in the archived traces** (README/commits only): FA4 host-D2H 0.78→1.0, host/routing sweep 0.879→1.15, page_size=1 misdiagnosis, local-memory ring arrays. Label these as documented outcomes, not as question→outcome evidence, unless the missing sessions are recovered.

**Kernels:**
- Hopper (`runs_h200`, 2026-10-30, 4 batches x 3 repeats on 8xH200): 182/182 correct, geomean 1.164× vs FA3 (1.162–1.164 across batches), 135/182 ≥1.0; vs fastest open-source 1.164 (FlashInfer never beats FA3; ours faster on 32/32); vs fastest available 1.096, 121/182 ≥1.0; cuDNN runs 145, ours faster on 110. claim set 1.005 (fp8 dense prefill bar 0.667); hd256_causal 0.94.
- Blackwell (`flash-cuda/docs/results_blackwell_*.csv`, recomputed): 124 FA4-paired scenarios, geomean 1.461×, 98 ≥1.0×, min 0.776.
  - attntest 1.44×; edge 2.02× (dominated by FA4 dispatch overhead, so flag it); claim 1.30×; real 1.10×.
  - The README quotes 1.161× / 1.412× and the gap analysis quotes 1.480×. **Pick one snapshot and regenerate.**

## 4. Experiments
**Before the Oct 30 deadline (feasible in ~2 days):**
1. Fig 1 data: Equip from `candidates.jsonl` (steps 0–132, timestamps); Ask from the 69fff5d7 trace (the geomean timeline is already extracted; assistant messages have no timestamps, so use the preceding human-turn time).
2. Score the baseline's `best.cu` on the 182-scenario Hopper suite (`bench.eval --target hopper --set all`). Minutes on the H200 node.
3. One fresh Blackwell `--set all` run to fix the headline snapshot. Kernel-only NCU ratios at locked clocks for the edge/attntest sets.
4. Second annotator for the 157 human messages (κ). About 2 hours.
5. Recover the Cursor model/version and, if possible, token/$ usage for the Ask sessions.

**For rebuttal / camera-ready (strongest answer to "a human did it"):**
6. Controlled ablation on a residual Blackwell gap (e.g. long-context dense or hd96), starting from the released kernel, with the same model and budget. Arms:
   - (A) Equip loop;
   - (B) bare agent + harness;
   - (C) bare agent + *automated* questioner drawing from the question bank at plateaus;
   - (D) human questioner.

   3 seeds if budget allows.
7. Non-expert questioner: someone without CUDA expertise steers using only the question bank.

## 5. Anticipated reviewer objections → where we answer
- "The human expert did the work." → §7 (discussion). The questions contain no CUDA; the agent wrote 100% of the code; about 71 tool calls per human message; automation ablation (exp. 6/7).
- "Unfair comparison (n=1, different front-ends/models, outage)." → §6 RQ2 and §7 threats. Same evaluator, configs, box and week; report the outage honestly; cross-score both kernels.
- "Only 25% of messages are questions." → §6 RQ3. Define Ask as interrogative steering and show the turning points were questions.
- "Speedups are host overhead, not kernels." → §6 RQ1. Report kernel-only ratios; the edge set is flagged.
- "Is this just AVO?" → §3.6 and §6 RQ4. We reuse AVO's loop; what changes is the source of directives. The last 22% came from a question, not more generations.

## 6. Data caveats found while drafting
- Baseline `meta.json` says effort xhigh, but it was high for 306 of 313 steps.
- The baseline evolution plot's flat tail is the auth outage, not the agent.
- `hd256_causal` measured 0.94× in the 2026-10-30 H200 runs (README's 0.843 was stale).
- Blackwell headline numbers disagree across README, the gap analysis and the CSVs.
- The `edge` set wins (about 2×) are mostly FA4 CuteDSL dispatch overhead.
- Run-detecting TMA depends on `arange` page tables; report the `FA4_NOPTRUN=1` numbers too.

## 7. Files
- `main.tex`, `sections/*.tex`: skeleton. The intro, KernelSeek (§3), the baseline section (§4), the case-study section (§5), the evaluation (§6) and related work (§8) are written; every other section has its contents as comments.
- `references.bib`: entries marked `TODO-authors` need completing.
- `analysis/`: scripts and `human_turns.jsonl`. The baseline scripts expect the checkpoint extracted at the path hard-coded inside them; re-point them before re-running.
- Build: `latexmk -pdf main.tex`. Set `\draftfalse` in `main.tex` to hide red TODOs.
