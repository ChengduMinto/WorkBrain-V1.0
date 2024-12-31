# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/12/2
"""
image_generation infer
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import cv2
import base64
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

app = FastAPI()

# 确保使用最佳性能
torch.backends.cudnn.benchmark = True

# 指定本地模型目录的路径
model_path = "./weight"#训练后的模型权重路径

# 使用 from_pretrained 方法加载本地模型
pipe = StableDiffusionPipeline.from_pretrained(
    model_path, 
    torch_dtype=torch.float16  # 根据您的硬件支持选择合适的数据类型
)

# 如果您使用GPU，请将模型移动到GPU上
if torch.cuda.is_available():
    pipe = pipe.to("cuda")

class Prompt(BaseModel):
    prompt: str

@app.post("/generate-image/")
async def generate_image(prompt: Prompt):
    try:
        # 使用模型生成图像
        image = pipe(prompt.prompt, guidance_scale=7.5).images[0]  

        # 保存图像到本地
        image.save('result.png')
        
        # 返回base64编码的字符串
        return {"image_path": './expert_models/image_generation/result.png'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 测试
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)