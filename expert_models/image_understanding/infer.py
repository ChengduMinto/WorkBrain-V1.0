# # -*- coding: utf-8 -*-
# # author: wjhan
# # date: 2024/11/7
# """
# image_generation infer
# """

import torch
from PIL import Image
from modelscope import AutoModel, AutoTokenizer

# #加载模型
model_dir =  "./expert_models/image_understanding/image_understanding_model"
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

def infer(img_path):
    image = Image.open(img_path).convert('RGB')
    question = '请描述以下这张图片的内容：'
    msgs = [{'role': 'user', 'content': [image, question]}]

    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )

    #print(answer)
    return answer

if __name__ == '__main__':
    img_path = 'test.jpg'
    img_res= infer(img_path)
    print(img_res)