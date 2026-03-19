from dataclasses import dataclass, field
from typing import Union


@dataclass
class SWAAConfig:
    sliding_window_size: Union[int, None] = field(
        default=None,
        metadata={"help": "Size of the sliding window for attention."}
    )
    keep_first: int = field(
        default=0,
        metadata={"help": "Number of tokens to keep at the beginning (sink tokens)."}
    )
    force_fa_decode: Union[bool, list] = field(
        default=False,
        metadata={
            "help": "Whether to enforce full attention decoding (bool). If a list, specifies layers for which to enforce full attention decoding."}
    )
    non_sliding_layers: list = field(
        default_factory=list,
        metadata={"help": "List of layer indices that do not use sliding attention (use full attention)."}
    )
    enable_linear_mem: bool = field(
        default=True,
        metadata={"help": "Whether to enable linear memory operations for attention computation."}
    )
    flash_attn_weight: float = field(
        default=0.9,
        metadata={"help": "Weight for flash attention output in hybrid attention (default: 0.9)."}
    )
    linear_mem_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for linear memory output in hybrid attention (default: 0.1)."}
    )
    linear_mem_mode: str = field(
        default='fused_recurrent',
        metadata={"help": "Linear memory operation mode. Options: 'fused_recurrent', 'fused_chunk', 'chunk' (default: fused_recurrent)."}
    )
    linear_mem_blend_mode: str = field(
        default='raw',
        metadata={"help": "How to blend linear memory output with flash attention. Options: 'raw', 'centered', 'orth', 'orth_match'."}
    )

    @property
    def mark(self):
        mark = ""

        if self.sliding_window_size is not None:
            assert self.sliding_window_size > 0, "sliding_window_size should be positive or None"
            # Format: swa=<window_size_k>k_sink=<keep_first>
            mark = "swa={}k_sink={}".format(int(self.sliding_window_size // 1000), self.keep_first)

            # Check if non_sliding_layers forms an arithmetic progression for shorter mark
            if len(self.non_sliding_layers) > 2:
                # Check step consistency
                try:
                    step = self.non_sliding_layers[1] - self.non_sliding_layers[0]
                    is_arithmetic = all(self.non_sliding_layers[i] - self.non_sliding_layers[i - 1] == step for i in
                                        range(2, len(self.non_sliding_layers)))
                except IndexError:
                    is_arithmetic = False

                if is_arithmetic:
                    # Format: _falayer=<count>_from<start>_step<step>
                    mark = mark + "_falayer={}_from{}_step{}".format(len(self.non_sliding_layers),
                                                                     self.non_sliding_layers[0], step)
                else:
                    # Format: _falayer=nonlazy<count>_from<start> (Generic non-arithmetic)
                    mark = mark + "_falayer=nonlazy{}_from{}".format(len(self.non_sliding_layers),
                                                                     self.non_sliding_layers[0])
            elif len(self.non_sliding_layers) > 0:
                 # Non-arithmetic or only 1-2 layers, use simple count
                 mark = mark + "_falayer={}_from{}".format(len(self.non_sliding_layers), self.non_sliding_layers[0])

            # Append force_fa_decode marker
            mark = (mark + "_fadec") if self.force_fa_decode else mark

        return mark

    def dict(self):
        """转换为字典（如果还没有这个方法）"""
        return {
            'sliding_window_size': self.sliding_window_size,
            'keep_first': self.keep_first,
            'force_fa_decode': self.force_fa_decode,
            'non_sliding_layers': list(self.non_sliding_layers),
            'enable_linear_mem': self.enable_linear_mem,
            'flash_attn_weight': self.flash_attn_weight,
            'linear_mem_weight': self.linear_mem_weight,
            'linear_mem_mode': self.linear_mem_mode,
            'linear_mem_blend_mode': self.linear_mem_blend_mode,
        }
