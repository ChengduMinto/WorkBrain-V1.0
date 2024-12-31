# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/11/7
"""
image_generation infer
"""
import cv2
import base64
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

# 确保使用最佳性能
torch.backends.cudnn.benchmark = True

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained("./weight", torch_dtype=torch.float16)#训练后的模型权重路径
pipe.to('cuda')

def infer(prompt):
    # 使用模型生成图像
    image = pipe(prompt, guidance_scale=7.5).images[0]  

    # 将PIL图像转换为NumPy数组
    image_np = np.array(image)
    
    # 将图像从RGB转换为BGR
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # 将图像保存到内存中的字节流
    _, buffer = cv2.imencode('.png', image_cv)
    
    # 将字节流编码为base64
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    # 保存图像到本地
    cv2.imwrite('result.png', image_cv)
    
    # 返回base64编码的字符串
    return img_str

if __name__ == '__main__':
    text = '中国山水画'
    res = infer(text)
    print(res)