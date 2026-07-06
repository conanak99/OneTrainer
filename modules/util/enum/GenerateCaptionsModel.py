from enum import Enum


class GenerateCaptionsModel(Enum):
    BLIP = 'BLIP'
    BLIP2 = 'BLIP2'
    QWEN3VL_4B = 'QWEN3VL_4B'
    WD14_VIT_2 = 'WD14_VIT_2'

    def __str__(self):
        return self.value
