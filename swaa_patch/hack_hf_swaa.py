import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from typing import Optional,Union
from functools import partial
import inspect
import os
import torch
import transformers
from transformers.models.qwen3.modeling_qwen3 import (Cache, apply_rotary_pos_emb, FlashAttentionKwargs, Unpack,
                                                      BaseModelOutputWithPast,
                                                      CausalLMOutputWithPast,Callable,ALL_ATTENTION_FUNCTIONS,eager_attention_forward)
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Attention,Qwen3DecoderLayer
from transformers.models.llama.modeling_llama import LlamaAttention,LlamaForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM,Qwen3MoeModel,Qwen3MoeAttention,MoeModelOutputWithPast,MoeCausalLMOutputWithPast,load_balancing_loss_func
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM,Qwen2Attention,Qwen2DecoderLayer
from transformers.utils import (
    logging,
)
from transformers.modeling_utils import flash_attention_forward,is_flash_attn_2_available
from transformers.modeling_flash_attention_utils import _process_flash_attention_kwargs,_hf_api_to_flash_mapping
from transformers.models.gemma2 import Gemma2Config
from transformers.models.gemma3 import Gemma3Config

from einops import rearrange, repeat
import torch.nn.functional as F
import torch.nn as nn
from fla.ops.linear_attn import chunk_linear_attn, fused_chunk_linear_attn, fused_recurrent_linear_attn
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from .swaa_config import SWAAConfig

# from .kernel.AnchorKernel import AnchorKernel,EluKernel
logger = logging.get_logger(__name__)

def _lazy_define_process_function_swaa(flash_function):
    """
    Depending on the version and kernel some features are not supported. Due to limitations in
    `torch.compile`, we opt to statically type which (optional) kwarg parameters are supported
    within `_process_flash_attention_kwargs`.

    NOTE: While all supported kwargs are marked as `True`, everything else is marked as `False`.
          This might be confusing for kwargs that we use in any case, e.g. `is_causal`.
    """

    flash_parameters = inspect.signature(flash_function).parameters
    process_parameters = inspect.signature(_process_flash_attention_kwargs_swaa).parameters

    supports_mapping = {}
    for param in process_parameters:
        fa_param = _hf_api_to_flash_mapping.get(param, param)
        supports_mapping[fa_param] = fa_param in flash_parameters

    return partial(_process_flash_attention_kwargs_swaa, supports_mapping=supports_mapping)

def _process_flash_attention_kwargs_swaa(
    query_length: int,
    key_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    s_aux: Optional[torch.Tensor] = None,
    keep_first: Optional[int] = None,
    force_fa_decode: Optional[bool] = None,
    supports_mapping: Optional[dict[str, bool]] = None,
    **kwargs,
):
    """
    Returns a set of kwargs that are passed down to the according flash attention function based on
    requested features and whether it is supported - depends on the version and kernel implementation
    which is dynamically configured at `lazy_import_flash_attention`. The (un)supported features can be
    inspected in `supports_mapping`, see `_lazy_define_process_function` for more details.

    Args:
        query_length (`int`):
            Length of the query states
        key_length (`int`):
            Length of the key states
        is_causal (`bool`):
            Whether we perform causal (decoder) attention or full attention.
        dropout (`float`):
            Attention dropout.
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to `1 / sqrt(head_dim)`.
        sliding_window (`int`, *optional*):
            The size of the sliding window, i.e. we look at a max of `sliding_window` tokens back.
        use_top_left_mask (`bool`):
            Deprecated behavior of older versions of flash attention requiring different masking.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
        s_aux (`torch.Tensor`, *optional*):
            Attention sink auxiliary that adds a `bias` to the attention calculation via an additional head.
        keep_first (`int`, *optional*):
            Number of initial tokens to always attend to when using sliding window attention.
        force_fa_decode (`bool`, *optional*):
            Whether to automatically only allow sliding window attention to be applied to the prefilling stage.
    Return:
        flash_kwargs (`dict`):
            A dict of kwargs that are requested and supported.
    """
    flash_kwargs = {
        "causal": is_causal and not (use_top_left_mask and query_length == 1),
        "softmax_scale": softmax_scale,
    }

    if supports_mapping["dropout_p"]:
        flash_kwargs["dropout_p"] = dropout

    if supports_mapping["window_size"] and sliding_window is not None and key_length > sliding_window:
        # The flash attention API sets inclusive boundaries, i.e. (4, 0) would take 4 tokens to the left
        # and the current token for a total size of 5. However, we usually define our window sizes by
        # their total window size (when causal). Encoder models as of now seldom use SWA and when they
        # do, they have a custom workaround (e.g. ModernBERT) which would align with this symmetric logic, i.e.
        # for a total of `2*sliding_window + 1`.
        flash_kwargs["window_size"] = (sliding_window - 1, sliding_window - 1)

    if supports_mapping["deterministic"]:
        flash_kwargs["deterministic"] = (
            deterministic if deterministic is not None else os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
        )

    if supports_mapping["softcap"] and softcap is not None:
        flash_kwargs["softcap"] = softcap

    # Only within kernel implementation atm
    if supports_mapping["s_aux"] and s_aux is not None:
        flash_kwargs["s_aux"] = s_aux

    # Get keep_first from kwargs and pass to flash_kwargs
    if supports_mapping["keep_first"] and keep_first is not None:
        flash_kwargs["keep_first"] = keep_first

    # Get force_fa_decode from kwargs and pass to flash_kwargs
    if supports_mapping["force_fa_decode"] and force_fa_decode is not None:
        flash_kwargs["force_fa_decode"] = force_fa_decode

    return flash_kwargs

def linear_mem_ops(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    last_state: torch.Tensor | None = None,
    use_cache: bool | None = False,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool | None = None,
    mode: str | None = None,
    kernel_q: Optional[nn.Module] = None,
    kernel_k: Optional[nn.Module] = None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        执行 SWAA 的 linear memory 分支计算，并返回当前步的线性读出结果和新的 recurrent state。

        这个函数是 flash attention 主分支之外的“全局线性记忆”路径，负责把
        当前层的 `q/k/v` 送入 FLA 的 linear attention kernel 中，得到一个
        与 softmax attention 输出同形状的补充分支结果，供后续融合使用。

        需要注意的几个关键约定：

        1. 输入张量布局
           调用方传入的 `q/k/v` 采用 Hugging Face attention 常见的布局：

               [batch, heads, seq, dim]

           而 `fla.ops.linear_attn` 系列 kernel 期望的布局是：

               [batch, seq, heads, dim]

           所以在真正调用 kernel 之前，这里会显式做一次 `transpose(1, 2)`。
           这一步很关键，否则状态会沿错误的维度累积。

        2. recurrent state 透传
           线性注意力在 decode 时依赖 recurrent state 作为历史压缩记忆。
           这里同时兼容两套命名：

           - `last_state` / `use_cache`
           - `initial_state` / `output_final_state`

           其中 `initial_state` 优先级更高；如果调用方显式传了它，就覆盖
           `last_state`。同理，`output_final_state` 如果显式给出，也会覆盖
           `use_cache`。这是为了和 `attention_forward_swaa()` 的调用接口保持
           一致，避免 decode 阶段 state 被吞掉。

        3. GQA 兼容
           如果模型使用 grouped-query attention，那么 `q` 的 head 数会大于
           `k/v` 的 head 数。为了匹配 FLA kernel 的逐 head 计算，这里会先把
           `k/v` 按组扩展到和 `q` 相同的 head 数，再送入 kernel。

        4. 可选 kernel 映射
           `kernel_q` 和 `kernel_k` 用来对 `q/k` 做特征映射，例如 NIAH、
           softplus、anchor 等实验里使用的 phi/kernel。`v` 不做变换。

        5. 返回值
           `o` 会在 kernel 输出后重新整理成：

               [batch, seq, hidden]

           其中 `hidden = heads * dim`，与主 attention 分支在输出投影前的形状
           对齐，便于后续直接和 flash attention 结果融合。

        参数：
            q:
                query 张量，形状为 `[B, Hq, T, D]`。
            k:
                key 张量，形状为 `[B, Hk, T, D]`。
            v:
                value 张量，形状为 `[B, Hk, T, D]`。
            attention_mask:
                可选的 padding mask，形状为 `[B, T_total]`。如果提供，会据此
                计算 `cu_seqlens` 以支持 unpadded kernel 路径。
            last_state:
                旧接口名，对应上一时刻的 recurrent state。
            use_cache:
                旧接口名，表示是否请求 kernel 返回新的 recurrent state。
            initial_state:
                新接口名，对应本次 kernel 的初始 recurrent state。
            output_final_state:
                新接口名，表示是否返回本次更新后的 recurrent state。
            mode:
                linear attention kernel 模式，支持 `fused_recurrent`、
                `fused_chunk`、`chunk`。若不指定，则短序列默认用
                `fused_recurrent`，长序列默认用 `fused_chunk`。
            kernel_q:
                可选的 query 特征映射模块。
            kernel_k:
                可选的 key 特征映射模块。
            **kwargs:
                额外参数，目前主要使用其中的 `cu_seqlens`。

        返回：
            一个二元组 `(o, recurrent_state)`：

            - `o`: 线性记忆分支输出，形状为 `[B, T, H*D]`
            - `recurrent_state`: 本次 kernel 计算后的新状态；如果没有请求输出，
              则可能为 `None`
        """
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        _, _, q_len, _ = q.shape  # 形状为 (batch_size, num_attention_heads, seq_length, head_dim)

        # 未显式指定 mode 时，短序列默认走 recurrent，长序列默认走 chunk。
        mode = mode if mode is not None else ('fused_recurrent' if q_len <= 64 else 'fused_chunk')

        cu_seqlens = kwargs.get('cu_seqlens')
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            # 这里暂不需要真的构造 unpadded hidden_states，只需要 cu_seqlens 即可。

        recurrent_state = initial_state if initial_state is not None else last_state
        output_final_state = bool(use_cache) if output_final_state is None else bool(output_final_state)

        if kernel_q is not None:
            q = kernel_q(q)
        if kernel_k is not None:
            k = kernel_k(k)

        # 兼容 GQA：把 k/v 扩展到和 q 相同的 head 数。
        # q 形状: (batch, num_heads, seq, head_dim)
        # k/v 形状: (batch, num_kv_heads, seq, head_dim)
        num_attention_heads = q.shape[1]
        num_kv_groups = num_attention_heads // k.shape[1]

        # 使用 repeat_interleave 获得更直接的内存布局，避免额外的 expand/reshape 组合。
        if num_kv_groups > 1:
            k = k.repeat_interleave(num_kv_groups, dim=1)
            v = v.repeat_interleave(num_kv_groups, dim=1)

        # FLA 的 linear attention kernel 期望输入布局为 [batch, seq, heads, dim]。
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        if mode == 'fused_recurrent':
            o, recurrent_state = fused_recurrent_linear_attn(
                q=q,
                k=k,
                v=v,
                initial_state=recurrent_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                normalize=True,
                scale= 1.0
            )
        elif mode == 'fused_chunk':
            o, recurrent_state = fused_chunk_linear_attn(
                q=q,
                k=k,
                v=v,
                initial_state=recurrent_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                normalize=True,
                scale = 1.0
            )
        elif mode == 'chunk':
            o, recurrent_state = chunk_linear_attn(
                q=q,
                k=k,
                v=v,
                initial_state=recurrent_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                normalize=True,
                scale = 1.0
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        o = rearrange(o, 'b t h d -> b t (h d)')
        return o, recurrent_state

def mem_proj(o_base: torch.Tensor, o_mem: torch.Tensor,eps = 1e-6) -> torch.Tensor:
        # 投影系数
        denom = (o_base * o_base).sum(dim=-1, keepdim=True).clamp_min(eps)
        coeff = (o_mem * o_base).sum(dim=-1, keepdim=True) / denom

        proj = coeff * o_base
        delta_mem = o_mem - proj
        return delta_mem

def blend_linear_output(
    o_base: torch.Tensor,
    o_mem: torch.Tensor,
    flash_attn_weight: float,
    linear_mem_weight: float,
    blend_mode: str = "raw",
    eps: float = 1e-6,
) -> torch.Tensor:
        """
        将 flash attention 分支输出与 linear memory 分支输出做融合。

        这个函数实际支持两大类策略：

        1. `raw` 直接混合
           把 `o_base` 和 `o_mem` 当成两条并列分支，按显式权重相加：

               mixed = flash_attn_weight * o_base + linear_mem_weight * o_mem

           这是 SWAA 最直接的融合方式，也保留了历史行为。
           但它的风险是：`o_mem` 的范数在实际运行中可能明显大于 `o_base`，
           因此即便 `linear_mem_weight` 看起来很小，也可能把最终分布拉偏。

        2. 残差注入模式：`centered`、`orth`、`orth_match`
           这几种模式不再把两条分支视作对等混合，而是把 flash attention
           输出视为“主干基线”，把 linear memory 输出先变换成一个更稳定的
           残差，再做小幅注入：

               mixed = o_base + linear_mem_weight * delta_mem

           在这些模式里，`flash_attn_weight` 会被有意忽略，因为目标不是重新
           加权两条路径，而是尽量保持预训练 softmax attention 的原始分布，
           只让 linear memory 提供一个受控的小修正。

        各模式语义如下：

        - `raw`
          直接做加权和，不做任何稳定化处理。

        - `centered`
          先减去 `o_mem` 在最后一维上的均值，消除通道上的整体偏置：

              delta_mem = o_mem - mean(o_mem)

          适合抑制“所有维度一起整体抬高/压低”的漂移。

        - `orth`
          去掉 `o_mem` 中与 `o_base` 平行的分量，只保留相对于 `o_base`
          的正交残差：

              delta_mem = o_mem - proj_{o_base}(o_mem)

          直觉上，如果 linear memory 只是顺着已有的 flash 输出方向继续放大，
          那通常更像是对语言分布的扰动，而不是有价值的新信息。

        - `orth_match`
          先像 `orth` 一样取正交残差，再把这个残差的范数匹配到 `o_base`
          的范数，最后再乘以外部系数 `linear_mem_weight`：

              delta_mem = o_mem - proj_{o_base}(o_mem)
              delta_mem = delta_mem * (||o_base|| / ||delta_mem||)

          这样一来，最终注入强度就主要由 `linear_mem_weight` 控制。
          例如 `linear_mem_weight=0.02` 时，注入量大致就是 base 范数的
          `0.02x`，通常比 `raw` 混合稳定得多，更适合短 prompt 生成。

        参数：
            o_base:
                flash attention 分支输出。形状应与输出投影前的 hidden state 一致。
            o_mem:
                linear memory 分支输出，形状必须与 `o_base` 相同。
            flash_attn_weight:
                仅在 `raw` 模式下生效，表示 flash attention 分支的权重。
            linear_mem_weight:
                在 `raw` 模式下表示 linear memory 分支权重；
                在非 `raw` 模式下表示残差注入强度。
            blend_mode:
                融合方式。支持 `raw`、`centered`、`orth`、`orth_match`。
            eps:
                防止投影和范数匹配时除零的小常数。

        返回：
            与 `o_base` 形状和 dtype 一致的融合结果张量。
        """
        if blend_mode == "raw":
            return o_base.mul(flash_attn_weight).add(o_mem, alpha=linear_mem_weight)

        base_f = o_base.float()
        mem_f = o_mem.float()
        delta_mem = mem_f

        if blend_mode == "centered":
            delta_mem = mem_f - mem_f.mean(dim=-1, keepdim=True)
        elif blend_mode == "orth":
            delta_mem = mem_proj(base_f, mem_f, eps=eps)
        elif blend_mode == "orth_match":
            delta_mem = mem_proj(base_f, mem_f, eps=eps)
            mem_norm = delta_mem.norm(dim=-1, keepdim=True).clamp_min(eps)
            base_norm = base_f.norm(dim=-1, keepdim=True).clamp_min(eps)
            delta_mem = delta_mem * (base_norm / mem_norm)
        else:
            raise ValueError(f"Unsupported linear_mem_blend_mode: {blend_mode}")

        # 非 raw 模式都属于“残差注入”：保留 flash 输出作为主干分布，
        # 只叠加一个小幅、经过稳定化处理的 linear memory 残差。
        mixed = base_f + linear_mem_weight * delta_mem
        return mixed.to(o_base.dtype)

def attention_forward_swaa(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache],
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:

    q_len = hidden_states.shape[1]
    batch_size = hidden_states.shape[0]
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Extract sliding_window_size, keep_first, force_fa_decode, non_sliding_layers from self.config.swaa_config
    swaa_config:SWAAConfig = self.config.swaa_config if hasattr(self.config, "swaa_config") else SWAAConfig()
    linear_kernel_q = self.config.kernel_q
    linear_kernel_k = self.config.kernel_k

    sliding_window_size=swaa_config.sliding_window_size
    non_sliding_layers=swaa_config.non_sliding_layers
    force_fa_decode=swaa_config.force_fa_decode
    keep_first=swaa_config.keep_first
    enable_linear_mem=swaa_config.enable_linear_mem
    flash_attn_weight=swaa_config.flash_attn_weight
    linear_mem_weight=swaa_config.linear_mem_weight
    linear_mem_mode=swaa_config.linear_mem_mode
    linear_mem_blend_mode=getattr(swaa_config, "linear_mem_blend_mode", "raw")

    # Disable sliding window if the current layer is in non_sliding_layers
    if int(self.layer_idx) in non_sliding_layers:
        sliding_window_size=None

    if isinstance(swaa_config.force_fa_decode,list):
        force_fa_decode= int(self.layer_idx) in swaa_config.force_fa_decode

    # Print SWAA configuration if environment variable SWAA_DEBUG is '1'
    if os.environ.get("SWAA_DEBUG", "0") == "1":
        print(
            "Attention initialized with sliding_window_size={}, keep_first={}, prefill_slide={}, non_sliding_layers={}".format(
                sliding_window_size, keep_first, force_fa_decode, non_sliding_layers
            ))


    # Get prompt_length from kwargs
    prompt_length = kwargs.get("prompt_length", None)
    if force_fa_decode and prompt_length is None and sliding_window_size is not None and self.training:
        raise ValueError("prompt_length must be provided in training when force_fa_decode=True and sliding_window_size is not None.")

    # shape (batch_size, num_attention_heads, seq_length, head_dim)
    if self.config.model_type in ["qwen3","qwen3_moe"]:
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    elif self.config.model_type in ["llama","qwen2"]:
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    else:
        raise ValueError("Unsupported model type: {}".format(self.config.model_type))

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    last_state = None

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # Save original k/v before cache update (only needed when lengths will differ)
        # This saves memory by only keeping a reference when necessary
        q_len = query_states.shape[2]
        key_states_for_linear = key_states
        value_states_for_linear = value_states

        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx,
                                                          cache_kwargs)

        # Use cache-updated k/v for linear attention only when lengths match
        if key_states.shape[2] == q_len:
            key_states_for_linear = key_states
            value_states_for_linear = value_states

        if past_key_values.is_recurrent_state_initialized(self.layer_idx):
            last_state = past_key_values.get_recurrent_state(self.layer_idx)
    else:
        key_states_for_linear = key_states
        value_states_for_linear = value_states


    if prompt_length is not None and force_fa_decode and sliding_window_size is not None:
        if batch_size > 1:
            raise NotImplementedError("batch size > 1 is not supported when force_fa_decode=True in training.")
        # Training phase with provided prompt_length and force_fa_decode=True:
        # Attention is calculated in two parts.
        # Part 1 (prefill) requires SWA; Part 2 (decoding) does not require SWA.
        # Split query_states
        query_states_prompt = query_states[:, :, :prompt_length, :]
        query_states_answer = query_states[:, :, prompt_length:, :]
        key_states_prompt = key_states[:, :, :prompt_length, :]
        value_states_prompt = value_states[:, :, :prompt_length, :]

        # Calculate attention for the prompt part using sliding window
        attn_output_prefill,_ = flash_attention_forward(
            self,
            query_states_prompt,
            key_states_prompt,
            value_states_prompt,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window_size,
            keep_first=keep_first,
            force_fa_decode=False,
            **kwargs,)

        # Calculate attention for the answer part using full attention
        attn_output_decode,_ = flash_attention_forward(
            self,
            query_states_answer,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=None,
            **kwargs,
        )

        # Concatenate outputs along sequence dimension (dim=1 after internal transpose)
        attn_output = torch.cat([attn_output_prefill, attn_output_decode], dim=1)

    else:
        # inference, or training without force_fa_decode
        attn_output, _ = flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window_size,
            keep_first=keep_first,
            force_fa_decode=force_fa_decode,
            **kwargs,
        )

    # ===================================================================#
    if enable_linear_mem:
        o,h = linear_mem_ops(
            self,
            q=query_states,
            k=key_states_for_linear,
            v=value_states_for_linear,
            initial_state=last_state,
            attention_mask=attention_mask,
            output_final_state=past_key_values is not None,
            mode=linear_mem_mode,
            kernel_q=linear_kernel_q,
            kernel_k=linear_kernel_k,
            **kwargs,
        )

        if past_key_values is not None and h is not None:
            past_key_values.state_update(h, self.layer_idx)

        o_linear = o
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = blend_linear_output(
            attn_output,
            o_linear,
            flash_attn_weight=flash_attn_weight,
            linear_mem_weight=linear_mem_weight,
            blend_mode=linear_mem_blend_mode,
        )
    else:
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    # =================================================================== #

    attn_output = self.o_proj(attn_output)
    return attn_output, None

def hybrid_causallm_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
) -> CausalLMOutputWithPast:

    # Get prompt_length
    prompt_length = kwargs.pop("prompt_length", None)
    # If labels are provided and prompt_length is None, calculate prompt_length:
    # prompt_length is the count of -100 in labels[0]
    if labels is not None and prompt_length is None:
        prompt_length = (labels[0] == -100).nonzero(as_tuple=False).shape[0] if labels[0].numel() > 0 else None

    if use_cache and past_key_values is None:
        print("Warning: use_cache is True but past_key_values is None.")

    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        prompt_length=prompt_length,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def hybrid_causallm_forward_moe(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
) -> MoeCausalLMOutputWithPast:

    # Get prompt_length
    prompt_length = kwargs.pop("prompt_length", None)
    # If labels are provided and prompt_length is None, calculate prompt_length:
    # prompt_length is the count of -100 in labels[0]
    if labels is not None and prompt_length is None:
        prompt_length = (labels[0] == -100).nonzero(as_tuple=False).shape[0] if labels[0].numel() > 0 else None

    if use_cache and past_key_values is None:
        print("Warning: use_cache is True but past_key_values is None.")

    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        cache_position=cache_position,
        prompt_length=prompt_length,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

    aux_loss = None
    if output_router_logits:
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

    return MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )


def hack_hf_swaa(training=False):
    if not is_flash_attn_2_available():
        raise ImportError("Flash Attention 2 is not available. "
                          "Please install customized flash-attn to use Sliding Window Attention Adaptation.")

    transformers.modeling_flash_attention_utils._process_flash_attention_kwargs=_process_flash_attention_kwargs_swaa
    #transformers.modeling_flash_attention_utils._lazy_define_process_function=_lazy_define_process_function_swaa


    Qwen3Attention.forward = attention_forward_swaa
    Qwen3MoeAttention.forward = attention_forward_swaa
    LlamaAttention.forward = attention_forward_swaa
    Qwen2Attention.forward= attention_forward_swaa

    if training:
        Qwen3ForCausalLM.forward = hybrid_causallm_forward
        Qwen3MoeForCausalLM.forward = hybrid_causallm_forward_moe
        LlamaForCausalLM.forward = hybrid_causallm_forward
        Qwen2ForCausalLM.forward = hybrid_causallm_forward

    print("Hacked Qwen3, Qwen3Moe, Qwen2, and Llama models to use customized attention for SWAA.")
