# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/10/22
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizer
from model import BertTextModel_last_layer, BertTextModel_encode_layer  # 模型模块
from config import parsers  # 配置模块
import json,os
import time

app = FastAPI()

# 加载配置
args = parsers()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 模型加载
def load_model(model_path, device, args):
    if args.select_model_last:
        model = BertTextModel_last_layer().to(device)
    else:
        model = BertTextModel_encode_layer().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 分类结果及置信度输出
def text_class_name(text, pred, args):
    results = torch.argmax(pred, dim=1).cpu().numpy().tolist()
    classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))

    # 获取每个类别的置信度，并保留两位小数
    probabilities = torch.softmax(pred, dim=1).cpu().detach().numpy().tolist()[0]
    result_json = {classification_dict[i]: round(prob, 2) for i, prob in enumerate(probabilities)}

    return {
        "text": text,
        "predicted_class": classification_dict[results[0]],
        "class_probabilities": result_json
    }

# 预测单个文本
def pred_one(args, model, device, text):
    tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)
    encoded_pair = tokenizer(text, padding='max_length', truncation=True, max_length=args.max_len, return_tensors='pt')
    token_ids = encoded_pair['input_ids'].to(device)
    attn_masks = encoded_pair['attention_mask'].to(device)
    token_type_ids = encoded_pair['token_type_ids'].to(device)

    all_con = (token_ids, attn_masks, token_type_ids)
    with torch.no_grad():
        pred = model(all_con)
        return text_class_name(text, pred, args)

# 加载模型
root, name = os.path.split(args.save_model_last)
save_best = os.path.join(root, str(args.select_model_last) + "_" + name)
model = load_model(save_best, device, args)

# 定义请求模型
class TextRequest(BaseModel):
    text: str  # 单个文本

# 定义预测接口
@app.post("/predict/")
async def predict(request: TextRequest):
    try:
        start = time.time()
        prediction = pred_one(args, model, device, request.text)
        end = time.time()
        prediction["time_taken"] = f"{end - start:.4f} s"
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)