from typing import Union, List, Tuple, Optional

from transformers import LlamaConfig

TargetPosType = Union[
    List[Union[List[int], Tuple[int, ...]]],
    Tuple[Union[List[int], Tuple[int, ...]], ...],
]


class LoKILlamaConfig(LlamaConfig):
    def __init__(self, target_pos: Optional[TargetPosType] = None, **kwargs):
        self.target_pos = target_pos  # 添加新参数
        super().__init__(**kwargs)
