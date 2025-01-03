# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/10/23

import os
import time
from tqdm import tqdm
from config import parsers
from utils import read_data, MyDataset
from torch.utils.data import DataLoader
from model import BertTextModel_encode_layer, BertTextModel_last_layer
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score

# 设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = "0,1,2"

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_sum, count = 0, 0
    for batch_index, batch_con in enumerate(train_loader):
        batch_con = tuple(p.to(device) for p in batch_con)
        pred = model(batch_con)

        optimizer.zero_grad()
        loss = loss_fn(pred, batch_con[-1].long())
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        count += 1

        if batch_index % 100 == 99 or batch_index == len(train_loader) - 1:
            msg = "[{0}/{1:5d}]\tTrain_Loss:{2:.4f}"
            print(msg.format(epoch + 1, batch_index + 1, loss_sum / count))
            loss_sum, count = 0.0, 0

def validate(model, device, dev_loader, save_best_path):
    global acc_min
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch_con in tqdm(dev_loader):
            batch_con = tuple(p.to(device) for p in batch_con)
            pred = model(batch_con)
            pred = torch.argmax(pred, dim=1)

            pred_label = pred.cpu().numpy().tolist()
            true_label = batch_con[-1].cpu().numpy().tolist()

            all_true.extend(true_label)
            all_pred.extend(pred_label)

    acc = accuracy_score(all_true, all_pred)
    print(f"dev acc: {acc:.4f}")

    if acc > acc_min:
        acc_min = acc
        torch.save(model.state_dict(), save_best_path)
        print(f"Best model saved with accuracy: {acc:.4f}")

if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_text, train_label = read_data(args.train_file)
    dev_text, dev_label = read_data(args.dev_file)

    train_data = MyDataset(train_text, train_label, with_labels=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    dev_data = MyDataset(dev_text, dev_label, with_labels=True)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=True)

    root, name = os.path.split(args.save_model_best)
    save_best_path = os.path.join(root, str(args.select_model_last) + "_" + name)
    root, name = os.path.split(args.save_model_last)
    save_last_path = os.path.join(root, str(args.select_model_last) + "_" + name)

    # 选择模型
    if args.select_model_last:
        model = BertTextModel_last_layer().to(device)
    else:
        model = BertTextModel_encode_layer().to(device)

    optimizer = AdamW(model.parameters(), lr=args.learn_rate)
    loss_fn = CrossEntropyLoss()

    acc_min = float("-inf")
    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, epoch)
        validate(model, device, dev_loader, save_best_path)

    model.eval()
    torch.save(model.state_dict(), save_last_path)
    print(f"Last model saved.")

    end = time.time()
    print(f"Total run time: {(end - start) / 60:.2f} minutes")