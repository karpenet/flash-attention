import math
from torch.nn import functional as F

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn_fwd(q, k, v):
    attn_scores = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    weighted_sum = attn_weights @ v

    return weighted_sum, attn_weights

def manual_attn_bwd(dout, q, k, v, attn_weights):
    d_attn_weights = dout @ v.transpose(-2, -1)
    dv = attn_weights.transpose(-2, -1) @ dout

    d_attn_score =  d_attn_weights * attn_weights * (1 - attn_weights)
    dq = d_attn_score @ k
    dk = d_attn_score.transpose(-2, -1) @ q

    return dq, dk, dv

