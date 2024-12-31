# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/11/27
# description: 模型推理逻辑
import requests,json

#文本生成 text_generation
def text_generation(query):

     # 构建请求参数
    url = "http://172.168.80.36:8002/infer/"  #  替换为您部署的接口地址
    data = {
        "text": query
    }

    # 发送POST请求
    response = requests.post(url, json=data)

    # 处理响应结果
    if response.status_code == 200:
        print("请求发送成功！")
        return response.json()
    else:
        print("请求发送失败！错误代码：", response.status_code)
        return None
    
#图片理解 image_understanding                
def image_understanding(img_content,query):

    # 构建请求参数
    url = "http://172.168.80.36:8004/image_understand/"  

    
    data = {"img_content":img_content,"query": query}
    
    # 发送POST请求
    response = requests.post(url,json=data)

    # 处理响应结果
    if response.status_code == 200:
        print("请求发送成功！")
        #return(response.text)
        return response.json()
    else:
        print("请求发送失败！错误代码：", response.status_code)
        
        
        
# 图片生成 image_generation                
def image_generation(query):

# 构建请求参数
    url = "http://172.168.80.36:8003/generate-image/"  

    data = {
        "prompt":query
    }

    # 发送POST请求
    response = requests.post(url, json=data)

    # 处理响应结果
    if response.status_code == 200:
        print("请求发送成功！")
        return response.json()
    else:
        print("请求发送失败！错误代码：", response.status_code)
        return None
    

 #语音识别 asr
def asr(audio_file,query):
    url = "http://172.168.80.36:9000/asr"

    files = {'audio_file': open(audio_file, 'rb')}

    data = {
            "initial_prompt": query
        }

    response = requests.post(url, files=files, data=data)
        
    files['audio_file'].close()

    # 处理响应结果
    if response.status_code == 200:
        print("请求发送成功！")
        return response.text
    else:
        print("请求发送失败！错误代码：", response.status_code)
        return None
    

 #语音合成  tts
def tts(query):

    url = "http://172.168.80.39:9880/tts"

    data = {
            "text":query,
            "text_language": "zh"
            } 

    response = requests.post(url, json=data)

    # 处理响应结果
    if response.status_code == 200:
        print("请求发送成功！")
        return response.content
    else:
        print("请求发送失败！错误代码：", response.status_code)
        return None

 #视频理解能力 video_understanding
def video_understanding(video,question):

    # 构建请求参数
    url = "http://36.170.52.76:7899/analyze_video"  

    files = {'video': open(video, 'rb')}

    data = {
            "question": question
        }

    response = requests.post(url, files=files, data=data)
        
    files['video'].close()

    # 处理响应结果
    if response.status_code == 200:
        print("请求发送成功！")
        dict_data = json.loads(response.text)

        return dict_data["answer"]
    else:
        print("请求发送失败！错误代码：", response.status_code)
        return None
        
    
 #文档问答 document_qa
def document_qa(file,question):
    url = "http://172.168.80.36:2230/ask"

    files = {'file': open(file, 'rb')}

    data = {
            "question": question
        }

    response = requests.post(url, files=files, data=data)
        
    files['file'].close()

    # 处理响应结果
    if response.status_code == 200:
        print("请求发送成功！")
        dict_data = json.loads(response.text)
        return dict_data["answer"]
    else:
        print("请求发送失败！错误代码：", response.status_code)
        return None
    
 #外部接口 external_services
def external_services(query):

     # 构建请求参数
    url = "http://172.168.80.36:8002/query/"  #  为实际的接口地址
    data = {
        "text": query
    }

    # 发送POST请求
    response = requests.post(url, json=data)

    # 处理响应结果
    if response.status_code == 200:
        print("请求发送成功！")
        return response.json()
    else:
        print("请求发送失败！错误代码：", response.status_code)
        return None
    

        
if __name__ == '__main__':
    file = "test.txt"
    question = "这个文档里写了什么内容"
    print(document_qa(file,question))
