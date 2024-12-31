# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/10/27
"""
workbrain_main2.py
"""

from config.config import tools
from model_gate_network.infer import gated_network  # 门控网络模型

# 根据概率选择模型工具并执行
def execute_high_probability_tools(tools, probabilities, input_text):
    if probabilities is None:
        probabilities = {}  # 如果probabilities是None，则初始化为空字典
    
    results = {}
    for tool_name, tool_func in tools.items():  # 遍历tools字典
        print(f"当前工具: {tool_name, tool_func}")  # 调试信息

        if tool_name in probabilities and probabilities[tool_name] > 0.3:  # 检查工具名称是否存在于概率字典中且概率大于0.3
            print(f"准备执行工具: {tool_name}")  # 调试信息
            try:
                # 调用工具函数并将input_text作为参数传递
                result = tool_func(input_text)
                results[tool_name] = (result, probabilities[tool_name])  # 存储结果和对应的概率
                print(f"工具 {tool_name} 执行成功，结果: {result}")  # 调试信息
            except Exception as e:
                print(f"执行工具 {tool_name} 时发生错误: {e}")
    return results

# 调用函数并加权聚合结果
def aggregate_results(tools, probabilities, input_text):
    if probabilities is None:
        print("没有计算出任何工具的概率分布。")
        return {}

    results = execute_high_probability_tools(tools, probabilities, input_text)
    if not results:
        print("没有工具的执行概率超过30%，或者所有工具执行时发生错误。")
    else:
        # 这里简单地将结果和权重打印出来，实际应用中可能需要更复杂的聚合逻辑
        for tool_name, (result, prob) in results.items():
            print(f"{tool_name}: {result} (权重: {prob})")
    return results

if __name__ == '__main__':
    # 示例输入
    input_text = "帮我写一篇《春天来了》的文章，并为这篇文章配一张图片."

    # 调用门控网络逻辑
    probabilities = gated_network(input_text, tools)
    # 打印概率分布
    print("概率分布：", probabilities)
    # 打印工具映射
    print("工具映射：", tools)
    # 执行聚合函数
    aggregate_results(tools, probabilities, input_text)