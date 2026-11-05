
import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


#### Information about the model and scenario ####
# Model: meta-llama/Llama-3.1-8B-Instruct
# Max sequence length (finetuning and inference): 8192
# LoRA rank: 16
# LoRA target modules: down_proj
# Max inference requests per batch: 256
# Max fwd tokens per batch: 256
# KV cache slots: 40000
# Data type: bfloat16
# 

num_layers=32
hidden_size = 4096
seq_len = 8192
data_size = 2
int_size = 4
vocab_size = 128256
max_tokens_per_batch = 256
intermediate_size = 14336

num_q_heads = 32
num_kv_heads = 8
kv_cache_slots = 40000
tot_num_heads = num_q_heads + num_kv_heads*2
head_dim = hidden_size // num_q_heads

input_size = seq_len*int_size

embed_weight_size = hidden_size*vocab_size*data_size
embed_output_size = seq_len*hidden_size*data_size
assert embed_output_size == 67108864
assert embed_weight_size == 1050673152


############### RMS Norm (1st layer) ###############
rms_norm_weight_size = hidden_size*data_size
rms_norm_meta_size = seq_len * data_size +seq_len*hidden_size*data_size

############### QKV Proj ###############
qkv_proj_output_size = seq_len*head_dim*tot_num_heads*data_size
qkv_proj_weight_size = head_dim*tot_num_heads*hidden_size*data_size
qkv_proj_linear_meta_size = 0

############### Attention ###############
# gqa_ptr_array_size = num_q_heads * 8
# qkv_max_proj_size = head_dim*tot_num_heads*max_tokens_per_batch * data_size
query_tmp_size = num_q_heads*head_dim*max_tokens_per_batch * data_size
complex_size_fwd = max_tokens_per_batch * head_dim * ((num_q_heads + num_kv_heads) / 2) * 8
# qk_prod_size = max_tokens_per_batch * seq_len * num_q_heads * data_size
inc_attn_meta = (
    query_tmp_size +
    complex_size_fwd
)
assert inc_attn_meta == 7340032
kv_cache = kv_cache_slots * head_dim * 2 * num_kv_heads * data_size
assert kv_cache == 163840000

peft_kv_cache_size = 2*num_kv_heads*head_dim*seq_len*data_size
complex_size_bwd = seq_len*head_dim*((num_q_heads + num_kv_heads) / 2) * 8
allocated_peft_buffer_size1 = seq_len*head_dim*num_q_heads*data_size
flash_attn_softmax_lse_size = seq_len*num_q_heads*4
flash_attn_out_size = seq_len*head_dim*num_q_heads*data_size
peft_token_infos_size = seq_len*12
peft_attn_meta = (
    peft_kv_cache_size +
    complex_size_bwd +
    allocated_peft_buffer_size1 +
    flash_attn_softmax_lse_size +
    flash_attn_out_size +
    peft_token_infos_size
)
assert peft_attn_meta == 336691200

############### OutProj ###############
out_proj_output_size = seq_len*hidden_size*data_size
out_proj_weight_size = hidden_size*head_dim*num_q_heads*data_size
out_proj_linear_meta_size = 0

############### Post-attention Residual RMS Norm ###############
post_attn_residual_rms_norm_weight_size = hidden_size*data_size
post_attn_residual_rms_norm_meta_size = (
    seq_len*4 +
    seq_len*hidden_size*data_size
)
assert post_attn_residual_rms_norm_meta_size == 67141632

############### Gate/Up Proj ###############
gate_up_proj_output_size = seq_len*intermediate_size*data_size
gate_up_proj_weight_size = intermediate_size*hidden_size*data_size
assert gate_up_proj_output_size == 234881024
assert gate_up_proj_weight_size == 117440512

############### SigmoidSiluMulti ###############
ssm_output_size = seq_len*intermediate_size*data_size
assert ssm_output_size == 234881024
ssm_meta_size = seq_len*2*intermediate_size*data_size
assert ssm_meta_size == 469762048

############## Down Proj ###############
down_proj_weight_size = hidden_size*intermediate_size*data_size
assert down_proj_weight_size == 117440512

############### LoRA ###############
lora_inference_size = 589824
lora_peft_size = 2*118784000

############### Residual RMS Norm (starting 2nd layer) ###############
layer2_residual_rms_norm_weight_size = hidden_size*data_size
layer2_residual_rms_norm_meta_size = seq_len*4 + seq_len*hidden_size*data_size
assert layer2_residual_rms_norm_meta_size == 67141632


############### Final RMS Norm ###############
final_rms_norm_weight_size = hidden_size*data_size
final_rms_norm_meta_size = seq_len*4 + seq_len*hidden_size*data_size
assert final_rms_norm_weight_size == 8192
assert final_rms_norm_meta_size == 67141632

############### LM Head ###############
lm_head_weight_size = hidden_size*vocab_size*data_size
lm_head_output_size = seq_len*vocab_size*data_size
assert lm_head_weight_size == 1050673152
assert lm_head_output_size == 2101346304


############### Decoding ###############
decoding_ouput_size = seq_len*vocab_size*data_size
decoding_activation_size = seq_len*vocab_size*data_size
assert decoding_ouput_size == 2101346304
assert decoding_activation_size == 2101346304


############### PEFT bwd gradients ###############
decoding_peft_bwd_input_size = 2101346304
fused_op_bwd_size = 32768 + 67108864*4 + 100663296 + 234881024*3

############### Total Memory Calculation ###############
# Sum only leaf variables (actual memory allocations, not parameters or intermediate variables)
total_memory = (
    input_size +
    embed_weight_size + embed_output_size +
    rms_norm_weight_size + rms_norm_meta_size +
    (num_layers-1)*(layer2_residual_rms_norm_weight_size + layer2_residual_rms_norm_meta_size) +
    (num_layers)*(
        qkv_proj_output_size + qkv_proj_weight_size + qkv_proj_linear_meta_size +
        inc_attn_meta +  # composite of query_tmp_size + complex_size_fwd
        kv_cache +
        peft_attn_meta +  # composite of multiple peft variables
        out_proj_output_size + out_proj_weight_size + out_proj_linear_meta_size +
        post_attn_residual_rms_norm_weight_size + post_attn_residual_rms_norm_meta_size +
        gate_up_proj_output_size + gate_up_proj_weight_size +
        ssm_output_size + ssm_meta_size +
        down_proj_weight_size +
        lora_inference_size + lora_peft_size
    ) + final_rms_norm_weight_size + final_rms_norm_meta_size +
    lm_head_weight_size + lm_head_output_size +
    decoding_ouput_size + decoding_activation_size +
    decoding_peft_bwd_input_size +
    fused_op_bwd_size
)

print(f"Total memory size: {total_memory:,} bytes")
print(f"Total memory size: {total_memory / (1000**3):.2f} GB")

print("\n" + "="*60)
print("Memory Usage by Operator Type:")
print("="*60)

# Linear operations
linear_memory = (
    num_layers * (
        qkv_proj_weight_size +
        out_proj_weight_size +
        2*gate_up_proj_weight_size +
        down_proj_weight_size
    ) + qkv_proj_output_size + out_proj_output_size + 2*gate_up_proj_output_size +
    lm_head_weight_size + lm_head_output_size
)
print(f"Linear Init Task: {linear_memory / (1000**3):.2f} GB")

# SigmoidSiluMulti operations
sigmoid_silu_memory = num_layers * (ssm_meta_size)
print(f"SigmoidSiluMultiMeta: {sigmoid_silu_memory / (1000**3):.2f} GB")

# Attention operations (PEFT)
peft_attention_memory = num_layers * peft_attn_meta
print(f"IncMultiHeadSelfAttentionMeta (peft): {peft_attention_memory / (1000**3):.2f} GB")

# LoRA finetuning
lora_finetuning_memory = num_layers * lora_peft_size
print(f"LoRA finetuning: {lora_finetuning_memory / (1000**3):.2f} GB")

# KV Cache
kv_cache_memory = num_layers * kv_cache
print(f"KV Cache: {kv_cache_memory / (1000**3):.2f} GB")

# RMS Norm operations
rms_norm_memory = (
    rms_norm_meta_size +
    (num_layers-1) * layer2_residual_rms_norm_meta_size +
    num_layers * post_attn_residual_rms_norm_meta_size +
    final_rms_norm_meta_size
)
print(f"ResidualRMSNormMeta: {rms_norm_memory / (1000**3):.2f} GB")

# Decoding operations
decoding_memory = decoding_activation_size
print(f"Decoding Init Task: {decoding_memory / (1000**3):.2f} GB")

# Decoding PEFT Backward
decoding_peft_bwd_memory = decoding_peft_bwd_input_size
print(f"Decoding PEFT Backward Task: {decoding_peft_bwd_memory / (1000**3):.2f} GB")

# Decoding Meta
decoding_meta_memory = decoding_activation_size 
print(f"DecodingMeta: {decoding_meta_memory / (1000**3):.2f} GB")

# Embedding operations
embedding_memory = embed_output_size + embed_weight_size
print(f"Embedding Init Task: {embedding_memory / (1000**3):.2f} GB")

# FusedOp PEFT Backward
fused_op_peft_bwd_memory = fused_op_bwd_size
print(f"FusedOp PEFT Backward Task: {fused_op_peft_bwd_memory / (1000**3):.2f} GB")

# Attention operations (inference)
inf_attention_memory = num_layers * inc_attn_meta
print(f"IncMultiHeadSelfAttentionMeta (inf): {inf_attention_memory / (1000**2):.2f} MB")

# SigmoidSiluMulti Init (output size only)
ssm_init_memory = ssm_output_size
print(f"SigmoidSiluMulti Init Task: {ssm_init_memory / (1000**2):.2f} MB")

# RMS Norm weights
rms_norm_weights_memory = (
    rms_norm_weight_size +
    (num_layers-1) * layer2_residual_rms_norm_weight_size +
    num_layers * post_attn_residual_rms_norm_weight_size +
    final_rms_norm_weight_size
)
print(f"Residual RMS Norm Init: {rms_norm_weights_memory / (1000**2):.2f} MB")

# RMSNormMeta (individual meta size)
rms_norm_meta_individual = rms_norm_meta_size
print(f"RMSNormMeta: {rms_norm_meta_individual / (1000**2):.2f} MB")

# rmsnorm_init_task (similar to RMSNormMeta)
rmsnorm_init_memory = rms_norm_meta_size
print(f"rmsnorm_init_task: {rmsnorm_init_memory / (1000**2):.2f} MB")

# LoRA inference
lora_inference_memory = num_layers * lora_inference_size
print(f"LoRA inference: {lora_inference_memory / (1000**2):.2f} MB")

# Constant Init Task (input size)
constant_init_memory = input_size
print(f"Constant Init Task: {constant_init_memory / (1000**1):.2f} KB")

# LinearMeta (assumed to be 0 as in terminal output)
linear_meta_memory = 0
print(f"LinearMeta: {linear_meta_memory}B")



buffer_memory = (
   qkv_proj_output_size + out_proj_output_size + 2*gate_up_proj_output_size + lm_head_output_size +
   kv_cache_memory +
   decoding_memory +
   decoding_peft_bwd_memory +
   embed_output_size +
   fused_op_peft_bwd_memory +
   inf_attention_memory +
   ssm_output_size + rms_norm_meta_size +
   lora_inference_memory +
   constant_init_memory +
   linear_meta_memory
)
activation_memory = num_layers * (
    ssm_meta_size + peft_attn_meta
) + rms_norm_memory + rms_norm_meta_size + decoding_meta_memory
gradient_memory = (
    num_layers * lora_peft_size
)
weights_memory = (
    embed_weight_size + 
    num_layers * (qkv_proj_weight_size + out_proj_weight_size + 2*gate_up_proj_weight_size + down_proj_weight_size) + lm_head_weight_size +
    rms_norm_weights_memory
)

print("\n" + "="*60)
print("Memory Breakdown by Type:")
print("="*60)
print(f"Buffer Memory: {buffer_memory / (1000**3):.2f} GB")
print(f"Activation Memory: {activation_memory / (1000**3):.2f} GB")
print(f"Gradient Memory: {gradient_memory / (1000**3):.2f} GB")
print(f"Weights Memory: {weights_memory / (1000**3):.2f} GB")

total_breakdown_memory = buffer_memory + activation_memory + gradient_memory + weights_memory
print("-" * 40)
print(f"Total (Breakdown): {total_breakdown_memory / (1000**3):.2f} GB")

# Create bar plots
plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# Plot 1: Memory breakdown by type (excluding buffer)
memory_types = ['Activation', 'Gradient', 'Weights']
memory_values = [
    activation_memory / (1000**3),
    gradient_memory / (1000**3), 
    weights_memory / (1000**3)
]

bars1 = ax1.bar(memory_types, memory_values, color=['#ff7f0e', '#2ca02c', '#1f77b4'])
ax1.set_ylabel('Memory (GB)')
ax1.set_title('Memory Breakdown by Type')
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars1, memory_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{value:.2f} GB', ha='center', va='bottom')

# Plot 2: Activation memory breakdown by operator
activation_operators = []
activation_values = []

# Break down activation memory by operator
ssm_meta_total = num_layers * ssm_meta_size
peft_attn_total = num_layers * peft_attn_meta
rms_norm_total = rms_norm_memory
rms_norm_meta_individual_mem = rms_norm_meta_size
decoding_meta_mem = decoding_meta_memory

activation_operators = ['SigmoidSiluMulti', 'Attention', 'RMS Norm', 'CrossEntropyLoss']
activation_values = [
    ssm_meta_total / (1000**3),
    peft_attn_total / (1000**3),
    (rms_norm_total / (1000**3)) + (rms_norm_meta_individual_mem / (1000**3)),
    decoding_meta_mem / (1000**3)
]

bars2 = ax2.bar(activation_operators, activation_values, 
                color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']) #'#9467bd'
ax2.set_ylabel('Memory (GB)')
ax2.set_title('Activation Memory Breakdown by Operator')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars2, activation_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{value:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('./memory_breakdown.pdf', dpi=300, bbox_inches='tight')

