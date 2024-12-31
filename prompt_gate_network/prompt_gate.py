# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/10/25
import json
import re
import sys
import os

# 假设这些导入和配置信息已经正确设置
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import SparkApi
from expert_models.experts import (
                            text_generation, image_understanding, video_understanding,image_generation, asr, tts, document_qa, external_services
                            )

tools = {
    "文本生成能力": text_generation,
    "以文生图能力": image_generation,
    "图片理解描述能力": image_understanding, 
    "视频理解能力": video_understanding, 
    "语音识别能力": asr,
    "语音合成能力": tts,
    "文档问答能力": document_qa,
    "外部接口能力": external_services
}

# Spark API配置信息
appid = "***"
api_secret = "***"
api_key = "***"
domain = "generalv3.5"
Spark_url = "wss://spark-api.xf-yun.com/v3.5/chat"

# 构建门控网络逻辑
def gated_network(instruction, tools):
    question = [{"role": "user", "content": f"根据指令 '{instruction}' 选择工具箱中的工具，将指令内容根据选择的工具提取输入内容，并给出每个工具对应的概率，概率为小数，所有工具对应的概率之和为1，生成格式为{{'工具名称1':[0.3,'输入内容'],...}}。已知工具名称包括：{', '.join(tools.keys())}。请确保输出是一个有效的JSON对象，键为工具名称，值为对应概率和输入内容组成的列表。例如指令是“帮我写一篇《春天来了》的文章，并为这篇文章生成一张图片。”，输出格式为：{{'文本生成能力':[0.3,'帮我写一篇《春天来了》的文章'],'以文生图能力':[0.7,'春天来了']}}。你只需要给出每个工具的名称、概率、输入内容，不要输出其他无关话术"}]

    # 调用SparkApi并获取响应
    SparkApi.answer = ""
    SparkApi.main(appid, api_key, api_secret, Spark_url, domain, question)
    
    tool_probabilities_str = SparkApi.answer
    
    # 清理字符串并确保是有效的JSON格式
    cleaned_str = re.sub(r"^```json|```$", "", tool_probabilities_str).strip()
    
    # 如果模型返回的是直接的键值对形式，将其转换为字典格式
    if not cleaned_str.startswith('{'):
        cleaned_str = '{' + cleaned_str + '}'
    
    # 确保字符串符合JSON格式，替换单引号为双引号，同时确保值也被双引号包围
    formatted_tool_probabilities_str = re.sub(r"':\s*$([^]]+)$", lambda m: '": [' + ','.join(f'"{item.strip()}"' for item in m.group(1).split(',')) + ']', cleaned_str)
    formatted_tool_probabilities_str = formatted_tool_probabilities_str.replace("'", '"').replace(',,', ',').replace('}{', '},{')
    
    try:
        probabilities = json.loads(formatted_tool_probabilities_str)
        
        # 验证返回的键是否都在工具集中
        if not all(key in tools for key in probabilities.keys()):
            raise ValueError("Not all keys in the probabilities match the available tools.")
        
        # 将列表转换回元组，以便后续处理保持一致性
        probabilities = {k: tuple(v) for k, v in probabilities.items()}
        
        return probabilities
    except json.JSONDecodeError as jde:
        print(f"JSON Decode Error: {jde}")
        print(f"Failed to parse: {formatted_tool_probabilities_str}")
        print(f"Original response from SparkApi: {SparkApi.answer}")  # 提供更多上下文信息
        return None
    except Exception as e:
        print(f"Error parsing probabilities: {e}")
        return None

if __name__ == '__main__':
    input_text = "帮我写一个带有图片的文章"
    
    probabilities = gated_network(input_text, tools)

    if probabilities is not None:
        print("\n工具及其概率:")
        for tool_name, (probability, instruction) in probabilities.items():
            print(f"{tool_name}: 概率: {probability:.2f}, 指令: {instruction}")
    else:
        print("未能获得有效的工具概率分布。")