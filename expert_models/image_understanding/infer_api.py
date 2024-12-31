# -*- coding: utf-8 -*-
"""
image_understanding infer_api with JSON input and Base64 encoded image
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer
import io
import base64

app = FastAPI()

# 加载模型
model_dir = "./image_understanding_model"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  # 指定使用哪块GPU，如cuda:0、cuda:1等
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,
                                  attn_implementation='sdpa', torch_dtype=torch.bfloat16)
model = model.eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# 定义请求体的数据模型
class ImageRequest(BaseModel):
    img_content: str = Field(..., title="Base64 Encoded Image", description="The content of the image as a base64 encoded string.")
    question: str = Field("请描述以下这张图片的内容：", title="Question", description="A question about the image.")

async def infer(image_request: ImageRequest):
    try:
        # 将base64编码的字符串转换为字节流
        img_bytes = base64.b64decode(image_request.img_content)
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # 构造消息列表
        msgs = [{'role': 'user', 'content': [image, image_request.question]}]

        # 模型预测
        answer = model.chat(
            image=None,  # 保持与示例代码一致
            msgs=msgs,
            tokenizer=tokenizer
        )

        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the request: {str(e)}")

@app.post("/image_understand/")
async def infer_route(image_request: ImageRequest):
    try:
        result = await infer(image_request)
        return result
    except HTTPException:
        raise
    except Exception as e:
        # 更具体的错误处理
        error_message = f"An error occurred while processing the request: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)