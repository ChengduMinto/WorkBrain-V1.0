# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/11/28
import os 
import requests

def transcribe_audio(audio_file_path, url, initial_prompt):
    """
    上传音频文件并转录。

    :param audio_file_path: 音频文件路径
    :param url: ASR 服务的 URL
    :param initial_prompt: 初始提示，传入中文或其他语言的字符串
    :return: 转录结果或错误信息
    """
    # 获取文件名
    audio_file_name = os.path.basename(audio_file_path)

    # 设置请求的参数
    params = {
        'encode': 'true',
        'task': 'transcribe',
        'initial_prompt': initial_prompt,
        'word_timestamps': 'false',
        'output': 'txt'
    }

    # 设置请求头
    headers = {
        'accept': 'application/json',
    }

    try:
        # 打开音频文件并构造 multipart/form-data 请求
        with open(audio_file_path, 'rb') as audio_file:
            files = {
                'audio_file': (audio_file_name, audio_file, 'audio/mpeg')
            }

            # 发送 POST 请求
            response = requests.post(url, headers=headers, params=params, files=files)

        # 检查响应状态码
        if response.status_code == 200:
            return response.text
        else:
            return f"请求失败，状态码：{response.status_code}\n响应内容：{response.text}"

    except FileNotFoundError:
        return f"文件未找到: {audio_file_path}"

    except Exception as e:
        return f"发生错误: {str(e)}"

# 示例调用
if __name__ == "__main__":
    # 示例参数
    audio_file_path = '/mnt/data2/minto/Workbrian_Open_Source_Project/zzw/audio_chat/英文音频.mp3'
    url = 'http://172.168.80.36:9000/asr'
    initial_prompt = '请你解析语音'

    result = transcribe_audio(audio_file_path, url, initial_prompt)
    print(result)
