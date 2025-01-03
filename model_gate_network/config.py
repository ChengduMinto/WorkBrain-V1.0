# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/10/28
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Bert model of argparse")
    parser.add_argument("--train_file", type=str, default=os.path.join("./data", "train.txt"), help="Path to the training data file")
    parser.add_argument("--dev_file", type=str, default=os.path.join("./data", "dev.txt"), help="Path to the development data file")
    parser.add_argument("--test_file", type=str, default=os.path.join("./data", "test.txt"), help="Path to the test data file")
    parser.add_argument("--classification", type=str, default=os.path.join("./data", "class.txt"), help="Path to the classification data file")
    parser.add_argument("--bert_pred", type=str, default="./bert-base-chinese", help="Path to the pre-trained BERT model")
    parser.add_argument("--select_model_last", type=bool, default=True, help="Select model BertTextModel_last_layer")
    parser.add_argument("--class_num", type=int, default=8, help="Number of classes")
    parser.add_argument("--max_len", type=int, default=38, help="Maximum length of sentences")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learn_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--filter_sizes", type=int, nargs='+', default=[2, 3, 4], help="Filter sizes for TextCNN")
    parser.add_argument("--num_filters", type=int, default=2, help="Number of filters for TextCNN")
    parser.add_argument("--encode_layer", type=int, default=12, help="Number of layers in Chinese BERT model")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size of BERT layers")
    parser.add_argument("--save_model_best", type=str, default=os.path.join("model", "best_model.pth"), help="Path to save the best model")
    parser.add_argument("--save_model_last", type=str, default=os.path.join("model", "last_model.pth"), help="Path to save the last model")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)