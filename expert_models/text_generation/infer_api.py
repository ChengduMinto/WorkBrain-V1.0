# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/11/28
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI()

# 模型路径
model_id = "./weight"#训练后的模型权重路径

# 加载模型
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda"
)

class Query(BaseModel):
    text: str

@app.post("/infer/")
async def infer(query: Query):
    try:
        # 对话
        messages = [
            {"role": "system", "content": "你是workbrain大模型，你会根据用户的问题给出回答。"},
            {"role": "user", "content": query.text},
        ]

        # 终止符
        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        # 生成回复
        outputs = pipe(
            messages,
            max_new_tokens=4096,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.9,
            top_p=0.9,
        )
        assistant_response = outputs[0]["generated_text"][-1]["content"]
        return {"response": assistant_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 测试
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)