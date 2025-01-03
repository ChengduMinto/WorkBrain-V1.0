# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/10/24

from config import parsers
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


def read_data(file):
    """
    读取文件，并返回文本和标签列表
    :param file: 数据文件路径
    :return: texts, labels
    """
    all_data = open(file, "r", encoding="utf-8").read().split("\n")
    texts, labels = [], []
    for data in all_data:
        if data:
            text, label = data.split("\t")
            texts.append(text)
            labels.append(label)
    return texts, labels


class MyDataset(Dataset):
    def __init__(self, texts, labels=None, with_labels=True):
        """
        自定义数据集类
        :param texts: 文本列表
        :param labels: 标签列表
        :param with_labels: 是否包含标签
        """
        self.all_text = texts
        self.all_label = labels
        self.max_len = parsers().max_len
        self.with_labels = with_labels
        self.tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)

    def __getitem__(self, index):
        text = self.all_text[index]

        # Tokenize the text to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(text,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.max_len,
                                      return_tensors='pt')  # Return torch.Tensor objects
        token_ids = encoded_pair['input_ids'].squeeze(0)
        attn_masks = encoded_pair['attention_mask'].squeeze(0)
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)

        if self.with_labels:  # True if the dataset has labels
            label = int(self.all_label[index])
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids

    def __len__(self):
        return len(self.all_text)


if __name__ == "__main__":
    # 示例数据路径
    train_text, train_label = read_data("data/train.txt")
    print(train_text[0], train_label[0])

    # 创建数据集和数据加载器
    trainDataset = MyDataset(train_text, labels=train_label, with_labels=True)
    trainDataloader = DataLoader(trainDataset, batch_size=3, shuffle=False)

    # 打印数据加载器中的数据
    for i, batch in enumerate(trainDataloader):
        print(batch[0], batch[1], batch[2], batch[3])