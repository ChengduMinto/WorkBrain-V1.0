# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/10/21
import json
import re
import sys
import os
import requests

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from expert_models.experts2 import (
    text_generation, image_understanding, image_generation, asr, tts, document_qa, external_services
)

tools = {
    "文本生成能力": text_generation,
    "以文生图能力": image_generation,
    "图片理解描述能力": image_understanding, 
    "语音识别能力": asr,
    "语音合成能力": tts,
    "文档问答能力": document_qa,
    "外部接口能力": external_services
}

# POST API配置信息
api_url = "http://localhost:8001/predict"

# 构建门控网络逻辑
def gated_network(instruction, tools):
    # 构建请求数据
    data = {"text":instruction}
    
    # 发送POST请求
    response = requests.post(api_url, json=data)
    
    # 检查响应状态
    if response.status_code != 200:
        print(f"请求失败，状态码: {response.status_code}")
        return None
    
    # 解析响应
    try:
        response_data = response.json()

    
        return response_data['prediction']['class_probabilities']

    except Exception as e:
        print(f"Error parsing probabilities: {e}")
        return None

if __name__ == '__main__':
    # 示例输入
    input_text = "帮我写一个带有图片的文章"

    # 调用门控网络逻辑
    probabilities = gated_network(input_text, tools)
    print(probabilities)
    if probabilities is not None:
        print("\n工具及其概率:")
        for tool_name, probability in probabilities.items():
            print(f"{tool_name}: {probability:.2f}")
    else:
        print("未能获得有效的工具概率分布。")