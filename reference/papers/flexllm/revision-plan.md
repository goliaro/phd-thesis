# Collie NSDI '26 Revision Todo List

**Paper**: "Collie: Token-Level Co-Serving of LLM Inference and FineTuning with SLO Guarantees" (#183)  
**Deadline**: September 23, 2025 (4 weeks before Oct 21 camera-ready deadline)

## 1. Clarifying and Justifying the Co-serving Motivation

- [x] Reframe the introduction to better articulate when co-serving is economically beneficial versus separate clusters and temporal/spatial sharing solutions
- [x] Reposition the RLHF/GRPO discussion as a future application of the engine rather than a current motivation
- [x] Remove or clarify that RLHF/GRPO workloads were not directly evaluated
- [x] Update models cited in intro

## 2. Addressing Evaluation Limitations

### New Baselines and Comparisons
- [x] Add a new, stronger baseline to compare Collie's performance to temporal sharing with dynamic frequency adjustment
- [x] Add an analytical comparison between DeepSpeed-MII, TensorRT-LLM, PipeSwitch and vLLM
- [x] Explicitly acknowledge why vLLM v1 + LlamaFactory were chosen as representative baselines

### Technical Clarifications
- [x] Mention the optimizations from suggested systems that have already been incorporated into vLLM v1/LlamaFactory (e.g., DeepSpeed ZeRO-3 optimization)
- [x] Explain why token-level finetuning is superior to a simple sequence-level parallelism approach
- [x] Clarify that inference workloads use real arrival patterns (BurstGPT data from production ChatGPT serving on Azure)
- [x] Explain that prompts were sampled from ShareGPT for comparability with concurrent works
- [x] Clarify that Collie targets offline finetuning workloads with batch size of 1
- [x] Document evaluation using original, unmodified dataset (Sky-T1) for real model finetuning

## 3. Discussing Memory Overhead and Technical Details

### Memory Analysis Expansion
- [x] Create a more detailed component-wise memory breakdown figure
- [x] Clearly explain how 80% memory savings are achieved
- [x] Explain how finetuning memory is preallocated according to chosen budget
- [x] Explain how inference memory (KV cache) is managed dynamically to minimize evictions
- [x] Explicitly quantify eviction rates (note: almost 0 in end-to-end experiments)

### Technical Validations
- [x] Add explicit discussion confirming that graph pruning doesn't affect model quality (gradients for frozen parameters are mathematically unnecessary)
- [x] Illustrate how scheduler can prevent tenant interference by adopting the Virtual Token Counter algorithm (Sheng et al., OSDI '24)

## 4. Analyzing Complexity vs. Benefits

- [x] Add new subsection with decision framework/table helping practitioners determine when Collie's complexity is warranted
- [x] Provide clear guidance that co-serving is most beneficial for variable workloads with daily fluctuations
- [x] Add explicit discussion of limitations (PEFT-only, etc.)
- [x] Quantify implementation complexity (9K LOC) and operational considerations
- [x] Add clearer comparison with simpler approaches (pre-emption based approaches, temporal sharing with dynamic frequency, etc.)

## 5. Additional Improvements

### Writing and Structure
- [ ] Fix all writing issues noted (incomplete figure captions, missing references, typos)
- [x] Restructure Section 2 and improve clarity throughout
- [x] Add a proper limitations section

### Technical Analysis
- [x] Strengthen the analytical discussion of how performance would degrade under stricter SLOs

## Timeline
**Target Completion**: September 23, 2025 (no later than 4 weeks before October 21 camera-ready deadline)