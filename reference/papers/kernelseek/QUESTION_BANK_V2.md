# Question bank v2: operator-agnostic, reference-free

Proposal for revising the 48-question bank (appendix A, `docs/questions.md`, skill "Ask yourself")
so the released agent works on any CUDA kernel, on any NVIDIA GPU, with or without a faster
implementation to compare against. Sources mined: siboehm SGEMM worklog (Ampere), Shankhdhar H100
GEMM worklog (Hopper), Modular Blackwell matmul series (4 parts), FA3 and FA4 papers, plus the 63
interrogative human messages in our traces.

## 1. Admission rules for a bank question

A question is admitted only if it satisfies all five:

1. **No embedded answer.** It names a *class* of hypothesis or a *quantity to measure*, never a
   technique to apply. "Is the non-MMA work overlapped with the MMAs or serialized by waits?" is
   admissible; "did you implement ping-pong softmax?" is not.
2. **Answerable by measurement with stock tools.** The expected answer is a number, a SASS fact,
   or a profiler field the agent can get from `ncu`, `cuobjdump`, `ptxas -v`, clock counters, or
   the harness. Opinions are not answers.
3. **Bound, not reference.** Where v1 said "the reference", v2 says "the bound": a roofline from
   peak unit throughputs, a cycle budget per tile, or, *if one exists*, another implementation.
   Questions that only make sense with a reference go in one clearly marked conditional group.
4. **Architecture as a parameter.** Generic questions say "this architecture's asynchronous MMA"
   or "the on-chip accumulator store"; the per-architecture checklists bind those to wgmma/TMEM/
   mma.sync. A generic question must make sense on Ampere, Hopper, and Blackwell.
5. **Operator content lives in plug-ins.** Nothing in the core names softmax, KV, paging, heads,
   causal masks, split-KV. The attention plug-in keeps those and is what the case study used.

## 2. What happens to the 48 existing questions

| Disposition | Count | Which |
|---|---|---|
| Keep as is | 10 | 1, 9, 14, 15, 21, 23, 41, 42, 43, 48 |
| Rewrite (drop "the reference", name the bound or make conditional) | 27 | 2–4, 6–8, 10–13, 16–20, 22, 25, 26, 32–35, 38, 39, 44, 46, 47 |
| Move to attention plug-in | 11 | 5, 24, 27–31, 36, 37, 40, 45 |

Patterns in the rewrites:

- **"Is the reference winning on X" → "is X above the bound".** Q3, Q6, Q19, Q20, Q32, Q33, Q34.
  Example: Q6 "are our active cycles still worse than the reference binary's?" becomes "at locked
  clocks with L2 flushed, how far are kernel cycles from the cycle budget you computed for this
  tile?"
- **"that the reference skips/amortizes/specializes" → "that could be skipped/cached/specialized".**
  Q11, Q12, Q13, Q16, Q17.
- **Attention-flavored enumerations → regime enumerations.** Q2 "long context, paged prefill,
  split decode, padded head dim" becomes "large problem, small problem, irregular extents, padded
  dimension". Q4 "key length / query length" becomes "the streamed dimension / the tiled
  dimension". Q44 "QK MMA, PV MMA, softmax" becomes "loads, each MMA, the non-MMA math, epilogue".
- **"Same as the reference" comparisons become conditional.** Q7, Q8 move to group K below.
- **Blackwell-only wording generalized where the mechanism is generic.** Q26 (divergent
  `tcgen05.wait`) becomes "any asynchronous wait or barrier under a divergent predicate" [G].

The eleven moved questions are not lost: each has a generic ancestor in the core (gather → "indirect
loads vs. contiguous"; split-KV → "split along the reduction dimension"; softmax ping-pong →
"non-MMA work overlapped or serialized"), and the plug-in keeps the attention instance.

## 3. The "Ask yourself" list in the skill

This list is the most reference-dependent text we ship: items 1, 2, 3, 5 and 6 all presuppose a
faster implementation to reverse-engineer. Proposed replacement (eight items, same role):

1. Which bound binds on this scenario (bandwidth, MMA, special-function unit, shared-memory
   bandwidth, issue slots, launch), and what fraction of it do we reach?
2. Is the gap in kernel cycles, or outside the kernel (host, launch, routing, harness)?
3. Which single unit or stage has the largest excess over its cycle budget, and what is my
   hypothesis for why?
4. Do the losing scenarios share one cause, and have I grouped them before fixing any?
5. Should the losing regime have its own kernel instead of a flag in a shared one?
6. Is every asynchronous mechanism this architecture offers actually asynchronous in my SASS,
   or are waits placed where they serialize?
7. Is the win I just measured larger than the noise of this GPU at this moment, and does it hold
   on the other regimes?
8. If another implementation of this operator exists on this machine, what does it do differently
   in SASS, in its launch path, and in its tile choices, and which of those explains my gap?

## 4. Proposed core bank (operator-agnostic)

Groups A–J are generic; K is conditional on a comparison implementation; L–N are per-architecture
checklists; O is the plug-in template. Tags: [G] generic, [H] Hopper, [B] Blackwell, [A] Ampere.
Questions are in the agent's voice. Where the source of the question is one of the worklogs the
tag in parentheses says which (SB = siboehm, PS = Shankhdhar H100, MO = Modular, FA3/FA4 = papers,
TR = our traces).

### A. Scope and triage (before anything else)
1. Which losing scenarios share a kernel body, and which only share a host launcher? (TR)
2. Do the losers form one regime (large, small, irregular extents, padded dimension) or a grab bag? (TR)
3. Which problem dimension does the gap grow with: the streamed one (feed), the tiled one (tiling, scheduling), or none (fixed tax)? (TR)
4. Is the gap in kernel cycles at locked clocks, or only in wall-clock (host, JIT, launch)? (TR)
5. Of the gaps and fixes I am listing, which have I already tried in this campaign, and why did they fail? (TR)
6. What do the scenarios below the target have in common, grouped by cause rather than by shape? (TR)

### B. Bound and regime (replaces "bound differencing")
7. From minimum bytes and FLOPs for this shape, which bound should dominate, and does measured time match it? (SB)
8. Am I memory-bound, compute-bound, latency-bound, or launch-bound on *this* scenario, and does the answer change across the suite? (SB, PS)
9. For my tile, what are the cycle counts per unit (MMA, special-function, FMA, shared memory, L2/HBM)? Which is largest? (FA4)
10. What is this GPU's MMA-to-special-function throughput ratio, and does my op mix exceed it? (FA3, FA4)
11. Is the peak I am measuring against the right one: locked clocks, the power state the benchmark runs at, this dtype's MMA rate? (PS)
12. Does shared-memory traffic scale with the number of MMA instructions rather than tile size, and is it the binder? (FA4)
13. Could power or thermal limits, not the kernel, be the ceiling at this size? (PS)

### C. Harness and measurement
14. Is the win larger than the noise measured *now* on this GPU, not the noise of an earlier run? (TR)
15. Am I timing what I think I am timing: warm-ups, L2 flush, same input distribution and dtype as the baseline, kernel-only vs. end-to-end? (SB, PS)
16. Is any of my time outside the kernel: metadata builds, combine passes, device-to-host copies, counter resets, descriptor builds? (TR)
17. Does the result hold on production-shaped problems, not only the square or power-of-two sizes I tuned on? (MO, PS)
18. Have I recorded clocks, power state, warm-up count, iteration count, and variance before claiming a percentage of peak? (MO)

### D. Host, launch, routing
19. Is there a synchronous device-to-host copy left on the losing route that another route already avoids? (TR)
20. Is work being redone per call on the host (descriptors, gather maps, schedules) that could be built once and cached? (TR, FA4)
21. Are there clears of counters, partial buffers, or outputs that nothing reads, or that could be fused? (TR)
22. For small grids, is the gap launch and setup overhead, with kernel-only time already near the bound? (TR)
23. Is this shape on the kernel I think it is, or on an older path because a routing gate is stale? (TR)
24. Would forcing the intended kernel on this shape, where numerically safe, close the gap without touching the kernel? (TR)
25. Is a feature flag or shape parameter forcing a slow instantiation that deserves its own specialization? (TR, SB: cuBLAS dispatches 16 kernels by shape)
26. For single-tile or underfilled grids, would a smaller tile or a split along the reduction dimension fill the machine? (TR, SB)

### E. Memory hierarchy and data movement
27. What is achieved global-memory throughput vs. peak, and does the profiler flag uncoalesced or sector-inefficient accesses? (SB)
28. Does the SASS show the widest loads and stores for every operand, or are some still 32-bit? (SB)
29. Where the compiler refuses to vectorize, can I prove alignment to it instead of unrolling by hand? (SB)
30. Are shared-memory bank conflicts reported for the exact access pattern the MMA reads, and does fixing them help end-to-end? (SB, MO: 8-way conflicts halved throughput)
31. What is the L2 hit rate, and which concurrently resident CTAs actually share input data? (SB, PS: 83% vs 70%)
32. Do neighboring CTAs load identical tiles that one load could serve (multicast, cluster sharing)? (PS, MO)
33. Are outputs written coalesced and vectorized through shared memory, with streaming hints where they are never re-read? (MO, PS)
34. Would a contiguous version of the same math meet the bound? If so, the gap is the indirect or gathered loads. (TR, generic ancestor of the paged-KV questions)
35. Do intermediates round-trip through shared or on-chip memory more than once between producers and consumers? (TR)
36. Before tuning the epilogue, have I confirmed the main loop is no longer bandwidth-bound? (MO: stmatrix epilogue gave 0% while memory-bound)

### F. Compute issue and instruction mix
37. What dominates the instruction mix: math, or address arithmetic and loads? What is the FLOPs-per-load ratio per thread? (SB)
38. Am I using the widest MMA shape the instruction set offers, and what does the wider one cost in registers and shared memory? (PS, FA4)
39. Does the compiler already do the caching, unrolling, or reordering I am about to hand-write? Check SASS first. (SB)
40. Is the non-MMA work (exp, conversions, reductions, rescaling) executing concurrently with MMAs, or serialized by waits? (FA3, FA4)
41. Did I confirm the intended overlap in SASS or a timeline, rather than assuming source order survived the compiler? (FA3 B.2)
42. Can part of a bottlenecked unit's work be re-expressed on an idle unit, and what fraction balances them? (FA4 exp emulation)
43. Which per-iteration bookkeeping is only occasionally necessary, and can it be made conditional while staying warp-uniform? (FA4 conditional rescale)
44. Am I doing per-element work (masking, bounds checks, predication) on tiles that are provably full or provably empty? (TR)
45. Do concurrent warps contend for the same slow unit at the same time, and should their critical sections be staggered? (FA3 ping-pong)
46. Are there type conversions or silent up-casts on the hot path? (TR)

### G. Occupancy, registers, shared memory
47. From registers, shared memory, and threads per block, what occupancy is achievable, and do stall reasons say more occupancy would help? (SB)
48. Is the binding limit the per-thread register cap rather than the register file, suggesting splitting work across more warps? (PS: 5× collapse at 256 accumulator regs)
49. Does ptxas warn about serialized MMAs or spills, and do I see local-memory traffic in SASS? (PS C7511)
50. Per warp role, what is the register budget for inputs, outputs, and temporaries, and does the sum fit? (FA4)
51. Are registers reallocated from issue-only warps to compute warps, and is the split tuned? (FA3, PS)
52. Can partial results be stored or consumed early to shorten live ranges? (FA4)
53. Did I size the multi-stage buffer from the shared-memory budget, or pick a stage count? Do stages and output buffers compete? (MO)
54. For dimensions that are not multiples of the MMA tile, am I on padded math, and is a native-width instantiation possible? (TR)
55. Does my thread-to-data mapping need cross-lane shuffles for reductions, or could another mapping remove them? (FA4)

### H. Asynchrony and pipelining
56. Measured in cycles, how long are the load, compute, and store phases per tile, and which is not overlapped? (PS: 1415 / 703 / 4572)
57. Have I drawn the dependency chain of one iteration and found which link leaves a unit idle? (MO: "half the hardware is idle")
58. Are data movement, MMA issue, and the non-MMA math in separate warps, or does one warp do all three in sequence? (FA3, MO)
59. How many independent MMAs are in flight, and is that enough to cover the non-MMA latency on this architecture? (FA4: at least two)
60. Are asynchronous operations committed without immediate waits, with the wait placed just before the consumer? (FA3)
61. Is the multistage pipeline at full depth on the losers, or draining early on partial tiles? (TR)
62. Which true dependence serializes the inner loop, and would one extra buffer let iteration j+1 overlap iteration j? (FA3)
63. Is the store path overlapped with the next tile's compute, or a serial tail? (MO, PS)
64. How many synchronizations per tile does the barrier API cost, and is the primitive itself the overhead? (PS: 257 → 3)
65. Is any asynchronous wait or barrier issued under a divergent predicate? (TR, generalized from Blackwell)
66. For a deeper pipeline I adopted or rejected, did I weigh its register cost against the forced tile-size reduction? (FA3 B.3)

### I. Scheduling and load balance
67. Does the tile count divide evenly into the SM count on the shapes that matter, or is there a tail wave? (MO: 256×224 → 148 CTAs; PS: 128 SMs beat 132)
68. In what order are tiles assigned to SMs, and would grouping or a space-filling curve raise L2 reuse? (PS, MO)
69. Would a persistent kernel help, and does its tile order preserve the locality the hardware scheduler gave me? (PS: persistent regressed 640 → 400)
70. Does the kernel pay launch, shared-memory, and barrier re-initialization per tile instead of per CTA lifetime? (MO)
71. Where variable extents or masks make tiles unequal, which tiles are the long tail, and are they issued first? (FA4 LPT)
72. For data-dependent work sizes, is the schedule computed once and cached? (FA4 varlen)
73. If work is split along the reduction dimension, is the split heuristic, the combine, or the partial-buffer traffic the cost? (TR, generic ancestor of split-KV)
74. Do contended global reductions run on dedicated warps, in an order that avoids stall chains? (FA3, FA4)

### J. Low-precision numerics
75. What operand layout does the low-precision MMA require, and where is the fix-up done and hidden? (FA3 fp8 transpose)
76. Is quantization-scale granularity matched to my tile so scaling rides on existing multiplies? (FA3)
77. What precision does the downstream consumer need, and did I measure error after that rounding against a high-precision reference? (FA4 Table 2)
78. For any approximation or skipped step, is there a final exact normalization that bounds the drift? (FA4)

### K. If another implementation of this operator exists (conditional)
79. Am I comparing against its steady-state binary, not a cold JIT compile or a first-call path? (TR)
80. Same buffers, dtype, flags, and split count as its call? (TR)
81. Which of its kernels is dispatched for this shape, and does it use a trick (split, specialization) I do not? (SB: Nsight Systems on cuBLAS)
82. On the same scenario, how do its unit utilizations and stall reasons differ from mine, and which difference matches my gap? (TR)
83. Is it constrained by the same rules as I am? If it is and it is faster, what does it do that I have ruled out? (TR: the FA3 harness question)

### L. Ampere-class checklist [A]
85. Am I on tensor-core MMA with `ldmatrix` operand loads, or on scalar FMA? (SB, PS)
86. Is global-to-shared staging asynchronous (`cp.async`) and double-buffered at both the global→shared and shared→register levels? (SB)
87. Are block, warp, and thread tiles chosen by autotuning on this GPU, and was the grid refiltered for alignment constraints? (SB: optimum not portable A6000 → A100)
88. Did the bank-conflict fix pay for itself, or did the layout change cost more than the conflicts? (SB: kernels 7–8 dropped)

### M. Hopper checklist [H]
89. Are MMAs the asynchronous warpgroup kind, with accumulators in registers budgeted per warpgroup? (PS, FA3)
90. Are tile loads done by the dedicated copy engine with a swizzle that matches the MMA's shared-memory layout? (PS)
91. Do producer warps release registers to consumers, and is the split tuned to the accumulator footprint? (PS, FA3)
92. Are neighboring CTAs in a cluster multicasting shared operands, and was cluster size tuned (2 beat 4)? (PS)
93. Are the barriers raw mbarriers with manual phase tracking, or a higher-level API that adds synchronizations? (PS)
94. Is the epilogue staged through shared memory and stored asynchronously, and does that trade against pipeline depth? (PS)

### N. Blackwell checklist [B]
95. Are MMAs fully asynchronous into tensor memory, issued by one thread, with completion signaled through barriers rather than waits? (MO, FA4)
96. Is tensor memory allocated once per CTA lifetime, and could its columns be split so the next tile accumulates while the last drains? (MO)
97. Is the accumulator or tensor-memory layout spilling or serializing MMA against the consumer warps? (TR)
98. Would a two-SM MMA pair halve per-SM operand traffic and shared-memory footprint for this tile? (MO, FA4)
99. Is only the leader CTA issuing MMAs, with peer barriers arrived at cluster scope? (MO)
100. Is a static tile scheduler assuming all SMs are free where the hardware launch-control scheduler could rebalance, and is scheduler latency hidden under the next load? (MO)
101. Is tensor-memory-to-register traffic off the critical path, and which register-resident fix-ups can move to a separate warp? (FA4)
102. Given the special-function unit did not scale with the MMA (16 vs 8192 ops/clk/SM), what fraction of that unit's work must move elsewhere? (FA4)

### O. Operator plug-in template (what the attention plug-in instantiates)
The plug-in supplies, for one operator: (i) its regimes (for attention: prefill/decode, dense/paged,
short/long, GQA/MQA); (ii) its irregular-work structure (causal, variable length); (iii) its
non-MMA units (softmax exp, rescale); (iv) its reduction splits (split-KV, combine); (v) its
indirect loads (page tables). The eleven moved questions (v1 Q5, 24, 27–31, 36, 37, 40, 45) become
this plug-in. The core questions already cover each via its generic ancestor (34, 40, 45, 71, 73).

## 5. Counts and what changes in the paper

| | v1 | v2 |
|---|---|---|
| Generic [G] | 32 (21 of which say "reference") | 78 (A–J), 0 presuppose a reference |
| Conditional on a comparison implementation | 0 (implicit everywhere) | 5 (K) |
| Architecture checklists | 3 [B] | 4 [A] + 6 [H] + 8 [B] |
| Operator plug-in | 13 [A]-tagged in the core | 11, outside the core |
| Total | 48 | 101 core + plug-in |

Text to update: §3.3 (counts, "organized by where the gap might be" → the A–O groups; the
"bound differencing" paragraph already says a roofline or speed-of-light serves the role, now the
bank enforces it); §3.5 the skill's list; H5 wording ("most questions contain nothing specific to
the operator or to a reference" becomes checkable: the core has zero); appendix A (full list with
source tags); `docs/questions.md` and `SKILL.md` in the release. Table 1's category names stay,
since they describe the human messages; only "bound differencing" gets a footnote that the bound
in the case study was usually the upstream kernel but need not be.

Honesty note for §6/RQ5: the v2 core is derived from the traces *and* from four public worklogs and
two papers, not from the traces alone. Say so; it strengthens the generality claim rather than
weakening it, since the sources are all GEMM or attention on three architectures and the questions
had to be stated once to cover all of them.

## 6. Categories I propose not adding

- **Per-operator categories in the core** (convolution, normalization, MoE routing). Each would
  need its own plug-in; the template in O is the mechanism, and we have only instantiated attention.
- **Per-GPU-SKU categories** (H100 vs H200, B200 vs GB200). SKU differences are numbers (SM count,
  bandwidth, power cap) that B.11 and B.13 ask the agent to look up, not question classes.
- **Backward-pass or multi-kernel pipelines.** Our evidence is forward-only. A later plug-in.
