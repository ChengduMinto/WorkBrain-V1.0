# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/11/28
import requests
from pydub import AudioSegment
from pydub.playback import play

def send_request_and_save_audio(text, output_path="output_audio.wav"):
    """
    发送 POST 请求到指定的 API，并保存返回的音频文件。
    
    :param text: 要发送的文本内容
    :param output_path: 保存音频的文件路径，默认为 output_audio.wav
    :return: 成功返回文件路径，失败返回错误信息
    """
    url = "http://172.168.80.39:9880"
    data = {
        "refer_wav_path": "/mnt/data/GPT-SoVITS/audio.wav",
        "prompt_text": "请将以下文本转为语音文字",
        "prompt_language": "zh",
        "text": text,
        "text_language": "zh"
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)

        if response.status_code == 200:
            # 将返回的音频内容保存到文件
            with open(output_path, "wb") as audio_file:
                audio_file.write(response.content)
            return output_path  # 返回文件路径
        else:
            return {
                "error": f"请求失败，状态码：{response.status_code}",
                "details": response.text
            }
    except requests.exceptions.RequestException as e:
        return {"error": "网络请求异常", "details": str(e)}

# 示例调用
if __name__ == "__main__":
    example_text = "先帝创业未半?中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。"
    audio_file_path = send_request_and_save_audio(example_text)
    
    if isinstance(audio_file_path, str):
        # 播放生成的音频文件
        audio = AudioSegment.from_file(audio_file_path)
        play(audio)
        print(f"音频已生成并保存到: {audio_file_path}")
    else:
        print(f"错误信息: {audio_file_path}")
