import torch
from typing import Dict, Optional, Tuple

def make_hash_params(
    *,
    num_heads: int,
    head_dim: int,
    r: int = 4,
    b: int = 10,
    eps: float = 1e-6,
    per_head_planes: bool = True,
    device=None,
    dtype=None,
    seed: int = 1234,
) -> Dict:
    """
    创建固定（不训练）的 SimHash 超平面参数。
    planes:
      - per_head_planes=True:  [H, R, b, D]
      - else:                 [R, b, D]
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    M = 1 << b
    if per_head_planes:
        planes = torch.randn(num_heads, r, b, head_dim, generator=g, device=device, dtype=dtype)
    else:
        planes = torch.randn(r, b, head_dim, generator=g, device=device, dtype=dtype)

    bit_weights = (2 ** torch.arange(b, device=device)).long()  # [b]

    return {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "r": r,
        "b": b,
        "M": M,
        "eps": eps,
        "per_head_planes": per_head_planes,
        "planes": planes,              # fixed
        "bit_weights": bit_weights,    # fixed
    }


def init_hash_state(
    batch_size: int,
    *,
    hash_params: Dict,
    device,
    dtype,
) -> Dict[str, torch.Tensor]:
    """
    初始化 hash 记忆状态（可放入 past_key_values 缓存）。
    S: [B,H,R,M,D]
    Z: [B,H,R,M]
    """
    H = hash_params["num_heads"]
    D = hash_params["head_dim"]
    R = hash_params["r"]
    M = hash_params["M"]
    S = torch.zeros(batch_size, H, R, M, D, device=device, dtype=dtype)
    Z = torch.zeros(batch_size, H, R, M, device=device, dtype=dtype)
    return {"S": S, "Z": Z}


def simhash_bucket_ids(x: torch.Tensor, *, hash_params: Dict) -> torch.Tensor:
    """
    x: [B,H,T,D] -> bucket: [B,H,R,T] in [0, M-1]

    支持 GQA (Grouped Query Attention)：
    - 如果 per_head_planes=True 且 x 的 head 数与 planes 不匹配，
      会自动 repeat x 以匹配 planes 的 head 数
    """
    planes = hash_params["planes"]
    bit_weights = hash_params["bit_weights"]
    per_head_planes = hash_params["per_head_planes"]

    if per_head_planes:
        # x [B,H,T,D], planes [H_pl,R,b,D] -> proj [B,H_pl,R,T,b]
        num_heads_in_planes = planes.shape[0]
        num_heads_in_x = x.shape[1]

        # Handle GQA: repeat x if head counts don't match
        if num_heads_in_x != num_heads_in_planes:
            if num_heads_in_planes % num_heads_in_x == 0:
                # x has fewer heads, repeat to match planes
                num_kv_groups = num_heads_in_planes // num_heads_in_x
                x = x.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1).reshape(
                    x.shape[0], num_heads_in_planes, x.shape[2], x.shape[3]
                )
            else:
                raise ValueError(
                    f"Cannot broadcast heads: x has {num_heads_in_x} heads, "
                    f"planes has {num_heads_in_planes} heads"
                )

        proj = torch.einsum("bhtd,hrkd->bhrtk", x, planes)
    else:
        # planes [R,b,D] -> proj [B,H,R,T,b]
        proj = torch.einsum("bhtd,rkd->bhrtk", x, planes)

    bits = (proj >= 0).to(torch.long)  # [B,H,R,T,b]
    bucket = torch.sum(bits * bit_weights.view(1, 1, 1, 1, -1), dim=-1)  # [B,H,R,T]
    return bucket


def hashmem_read_block(
    *,
    buckets_q: torch.Tensor,        # [B,H,R,L]
    state: Dict[str, torch.Tensor], # S,Z
    eps: float,
) -> torch.Tensor:
    """
    从 state 读取：o_block [B,H,L,D]
    """
    S = state["S"]  # [B,H,R,M,D]
    Z = state["Z"]  # [B,H,R,M]
    B, H, R, L = buckets_q.shape
    D = S.shape[-1]

    # gather S: [B,H,R,L,D]
    idxS = buckets_q.unsqueeze(-1).expand(-1, -1, -1, -1, D)  # [B,H,R,L,D]
    Sg = torch.gather(S, dim=3, index=idxS)                   # [B,H,R,L,D]

    # gather Z: [B,H,R,L]
    Zg = torch.gather(Z, dim=3, index=buckets_q)              # [B,H,R,L]

    o = Sg / (Zg.unsqueeze(-1) + eps)                         # [B,H,R,L,D]
    return o.mean(dim=2)                                      # average over R -> [B,H,L,D]


def hashmem_write_block_functional(
    *,
    buckets_k: torch.Tensor,         # [B,H,R,L]
    v_block: torch.Tensor,           # [B,H,L,D]
    state: Dict[str, torch.Tensor],  # old state
    weight: Optional[torch.Tensor] = None,  # [B,H,L] nonneg
) -> Dict[str, torch.Tensor]:
    """
    函数式写入：返回 new_state（不 in-place 修改输入 state）。
    更新规则：对每个表 r，在桶 buckets_k 写入 v_block（按 weight 加权）。
    """
    S = state["S"]
    Z = state["Z"]
    B, H, R, L = buckets_k.shape
    D = v_block.shape[-1]

    if weight is None:
        w = torch.ones(B, H, L, device=v_block.device, dtype=v_block.dtype)
    else:
        w = torch.clamp(weight.to(v_block.dtype), min=0)

    # 展开到每个表
    vR = v_block.unsqueeze(2).expand(-1, -1, R, -1, -1)              # [B,H,R,L,D]
    wR = w.unsqueeze(2).expand(-1, -1, R, -1)                         # [B,H,R,L]
    addS = vR * wR.unsqueeze(-1)                                      # [B,H,R,L,D]
    addZ = wR                                                          # [B,H,R,L]

    # functional: clone then scatter_add_
    S_new = S.clone()
    Z_new = Z.clone()

    idxS = buckets_k.unsqueeze(-1).expand(-1, -1, -1, -1, D)          # [B,H,R,L,D]
    S_new.scatter_add_(dim=3, index=idxS, src=addS)                   # scatter along M
    Z_new.scatter_add_(dim=3, index=buckets_k, src=addZ)

    return {"S": S_new, "Z": Z_new}


def block_causal_hashmem(
    *,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,               # [B,H,T,D]
    state: Dict[str, torch.Tensor],
    hash_params: Dict,
    block_size: int = 256,
    weight: Optional[torch.Tensor] = None,                            # [B,H,T] nonneg
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    block-causal 一次性 16k：按块执行
      - 读：使用块开始前的 state
      - 写：将整块写入更新 state
    返回：
      o: [B,H,T,D]
      new_state

    支持 GQA: 如果 k/v 的 head 数与 q 不同，会自动扩展以匹配

    注意：调用方应确保 k/v 的序列长度与 q 相同。
    """
    B, H_q, T_q, D = q.shape
    T_k = k.shape[2]
    T_v = v.shape[2]
    H_k = k.shape[1]
    H_v = v.shape[1]
    eps = hash_params["eps"]

    # Handle GQA: expand k, v to match q's number of heads
    # 注意：使用各自的序列长度 T_k, T_v
    if H_k != H_q:
        num_kv_groups = H_q // H_k
        if H_q % H_k == 0:
            k = k.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1).reshape(B, H_q, T_k, D)
        else:
            raise ValueError(f"Cannot broadcast k heads: q has {H_q}, k has {H_k}")

    if H_v != H_q:
        num_kv_groups = H_q // H_v
        if H_q % H_v == 0:
            v = v.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1).reshape(B, H_q, T_v, v.shape[-1])
        else:
            raise ValueError(f"Cannot broadcast v heads: q has {H_q}, v has {H_v}")

    # Expand weight if needed
    if weight is not None and weight.shape[1] != H_q:
        H_w = weight.shape[1]
        T_w = weight.shape[2]
        num_kv_groups = H_q // H_w
        if H_q % H_w == 0:
            weight = weight.unsqueeze(2).expand(-1, -1, num_kv_groups, -1, -1).reshape(B, H_q, T_w)
        else:
            raise ValueError(f"Cannot broadcast weight heads: q has {H_q}, weight has {H_w}")

    buckets_q = simhash_bucket_ids(q, hash_params=hash_params)  # [B,H,R,T_q]
    buckets_k = simhash_bucket_ids(k, hash_params=hash_params)  # [B,H,R,T_k]

    outs = []
    cur_state = state

    for start in range(0, T_q, block_size):
        end = min(T_q, start + block_size)
        L = end - start

        bq_blk = buckets_q[:, :, :, start:end]  # [B,H,R,L]
        bk_blk = buckets_k[:, :, :, start:end]  # [B,H,R,L]
        v_blk = v[:, :, start:end, :]           # [B,H,L,D]
        w_blk = None if weight is None else weight[:, :, start:end]  # [B,H,L]

        # read with state BEFORE this block
        o_blk = hashmem_read_block(buckets_q=bq_blk, state=cur_state, eps=eps)  # [B,H,L,D]
        outs.append(o_blk)

        # write whole block -> new state
        cur_state = hashmem_write_block_functional(
            buckets_k=bk_blk, v_block=v_blk, state=cur_state, weight=w_blk
        )

    o = torch.cat(outs, dim=2)  # [B,H,T,D]
    return o, cur_state

