# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/10/23
from model import BertTextModel_last_layer
from utils import MyDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
import json  # 导入json模块
from config import parsers
import time
import os


def load_model(model_path, device, args):
    if args.select_model_last:
        model = BertTextModel_last_layer().to(device)
    else:
        model = BertTextModel_encode_layer().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def text_class_name(text, pred, args):
    results = torch.argmax(pred, dim=1).cpu().numpy().tolist()
    classification = open(args.classification, "r", encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)), classification))

    # 获取每个类别的置信度，并保留两位小数
    probabilities = torch.softmax(pred, dim=1).cpu().detach().numpy().tolist()[0]
    result_json = {classification_dict[i]: round(prob, 2) for i, prob in enumerate(probabilities)}

    print(f"文本：{text}\t预测的类别为：{classification_dict[results[0]]}")
    print("各分类置信度：")
    print(json.dumps(result_json, ensure_ascii=False, indent=4))  # 输出JSON格式的结果


def pred_one(args, model, device, start, text):
    tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)
    encoded_pair = tokenizer(text, padding='max_length', truncation=True, max_length=args.max_len, return_tensors='pt')
    token_ids = encoded_pair['input_ids']
    attn_masks = encoded_pair['attention_mask']
    token_type_ids = encoded_pair['token_type_ids']

    all_con = tuple(p.to(device) for p in [token_ids, attn_masks, token_type_ids])
    pred = model(all_con)
    text_class_name(text, pred, args)
    end = time.time()
    print(f"耗时为：{end - start} s")


def infer(text):
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    root, name = os.path.split(args.save_model_last)
    save_best = os.path.join(root, str(args.select_model_last) + "_" + name)
    model = load_model(save_best, device, args)

    print("模型预测结果：")
    start = time.time()
    pred_one(args, model, device, start, text)


if __name__ == "__main__":
    text = "给我画一张图"
    infer(text)