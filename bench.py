import torch
from torch.utils.cpp_extension import load
from baseline import *

# Load the CUDA kernel as a python module
minimal_attn = load(name='minimal_attn', sources=['main.cpp', 'flash.cu'], extra_cuda_cflags=['-O2'])

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 16
n_head = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
dout = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

print('=== profiling manual attention ===')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result_fwd, attention_weights = manual_attn_fwd(q, k, v)
    manual_result_bwd = manual_attn_bwd(dout, q, k, v, attention_weights)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('=== profiling minimal flash attention === ')

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    minimal_result = minimal_attn.forward(q, k, v)
print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))

print('attn values sanity check:', torch.allclose(minimal_result, manual_result_fwd, rtol=0, atol=1e-02))
