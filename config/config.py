# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/10/28
from expert_models.experts2 import (
                            text_generation, image_understanding, video_understanding,image_generation,asr,tts,document_qa,external_services
                            )

# 专家模型集合
tools = {
    "文本生成能力": text_generation,
    "以文生图能力": image_generation,
    "图片理解能力": image_understanding, 
    "视频理解能力": video_understanding, 
    "语音识别能力": asr,
    "语音合成能力": tts,
    "文档问答能力": document_qa,
    "外部接口能力": external_services
}
