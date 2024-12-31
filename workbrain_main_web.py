# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/10/28
import gradio as gr
import os, sys
import base64
from PIL import Image
import io
import asyncio
from config.config import tools  # tools 是一个包含工具函数的字典
from prompt_gate_network.prompt_gate import gated_network  # 提示工程

# 保留原有的路径设置和导入
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

async def create_upload_file(img: Image.Image) -> str:
    try:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"图像转换失败: {e}")
        raise

async def execute_high_probability_tools(tools, probabilities_and_prompts):
    try:
        if not probabilities_and_prompts:
            return {}
        
        results = {}
        for tool_name, (probability, prompt) in probabilities_and_prompts.items():
            print(f"当前工具: {tool_name}")

            if probability > 0.3:
                print(f"准备执行工具: {tool_name}，使用提示: {prompt}")
                tool_func = tools.get(tool_name)
                if tool_func:
                    result = None
                    if tool_name == "文本生成能力":
                        text_prompt = prompt.get('text', '')
                        result = tool_func(text_prompt)
                    elif tool_name == "以文生图能力":
                        text_to_image = prompt.get('text')
                        result = tool_func(text_to_image)
                    elif tool_name == "图片理解能力":
                        image = prompt.get('image')
                        text = prompt.get('text', '请描述以下这张图片的内容：')
                        if image and isinstance(image, Image.Image):
                            img_content = await create_upload_file(image)
                            result = tool_func(img_content, text)
                            if asyncio.iscoroutine(result): 
                                result = await result
                    elif tool_name == "视频理解能力":
                        video_path = prompt.get('video')
                        text = prompt.get('text', '请分析这段视频内容')
                        if video_path and os.path.exists(video_path):
                            result = tool_func(video_path, text)
                            if asyncio.iscoroutine(result): 
                                result = await result
                    elif tool_name == "语音识别能力":
                        audio_path = prompt.get('audio')
                        text = prompt.get('text', '请识别这段语音')  
                        if audio_path and os.path.exists(audio_path):
                            with open(audio_path, 'rb') as audio_file:
                                result = tool_func(audio_file.name, text)
                                if asyncio.iscoroutine(result): 
                                    result = await result
                    elif tool_name == "语音合成能力":
                        text_to_speak = prompt.get('text')
                        audio_bytes = tool_func(text_to_speak)
                        if audio_bytes:
                            result = {"audio_bytes": audio_bytes}
                        else:
                            print("语音合成失败")
                        if asyncio.iscoroutine(result): 
                            result = await result
                    elif tool_name == "文档问答能力":
                        doc_path = prompt.get('document')
                        question = prompt.get('text', '请问该文档中提到的关键信息是什么？')
                        if doc_path and os.path.exists(doc_path):
                            result = tool_func(doc_path, question)
                            if asyncio.iscoroutine(result): 
                                result = await result
                    elif tool_name == "外部接口能力":
                        api_params = prompt.get('api_params', {})
                        result = tool_func(api_params)
                        if asyncio.iscoroutine(result): 
                            result = await result

                    if result is not None:
                        results[tool_name] = (result, probability)
                        print(f"工具 {tool_name} 执行成功，结果: {result}")
        return results
    except Exception as e:
        print(f"工具执行失败: {e}")
        raise

def aggregate_results(tools, probabilities_and_prompts):
    try:
        if not probabilities_and_prompts:
            print("没有计算出任何工具的概率分布。")
            return {}

        results = asyncio.run(execute_high_probability_tools(tools, probabilities_and_prompts))
        if not results:
            print("没有工具的执行概率超过30%，或者所有工具执行时发生错误。")
        else:
            for tool_name, (result, prob) in results.items():
                print(f"{tool_name}: {result} (权重: {prob})")
        return results
    except Exception as e:
        print(f"聚合结果失败: {e}")
        raise

def process_input(input_text, input_file=None):
    try:
        input_data = {'text': input_text}

        if input_file is not None:
            file_extension = os.path.splitext(input_file.name)[1].lower()
            if file_extension in ['.png', '.jpg', '.jpeg']:
                with open(input_file.name, 'rb') as image_file:
                    with Image.open(image_file) as img:
                        input_data['image'] = img.convert('RGB')
                        input_data['text'] = f"{input_text}\n请描述以下这张图片的内容："
            elif file_extension in ['.wav', '.mp3']:
                input_data['audio'] = input_file.name
                input_data['text'] = '请识别这段语音'
            elif file_extension in ['.mp4', '.avi']:
                input_data['video'] = input_file.name
                input_data['text'] = '请识别这段视频'
            elif file_extension in ['.docx', '.txt','.pptx', '.pdf', '.xlsx']:
                input_data['document'] = input_file.name
                input_data['text'] = '这个文档里面写了什么内容？'

        gated_output = gated_network(input_data['text'], tools)

        if gated_output is None:
            gated_output = {}

        probabilities_and_prompts = {
            tool_name: (float(prob), {'text': instruction, 'image': input_data.get('image'), 'audio': input_data.get('audio'),'video': input_data.get('video'),'document': input_data.get('document')})
            for tool_name, [prob, instruction] in gated_output.items()
        }

        text_and_image_output = aggregate_results(tools, probabilities_and_prompts)

        outputs = {
            'text': '',
            'image': None,
            'audio': None,
            'video': None,
            'document':None,
            'show_text': False,
            'show_image': False,
            'show_audio': False,
            'show_video': False,
            'show_document': False
        }

        for tool_name, (result, prob) in text_and_image_output.items():
            if isinstance(result, dict):
                if 'response' in result:  
                    outputs['text'] += "\n" + result['response']
                    outputs['show_text'] = True
                if 'image_path' in result and os.path.exists(result['image_path']):
                    with open(result['image_path'], 'rb') as f:
                        image_bytes = f.read()
                        outputs['image'] = Image.open(io.BytesIO(image_bytes))
                        outputs['show_image'] = True
                if 'audio_bytes' in result: 
                    outputs['audio'] = result['audio_bytes']
                    outputs['show_audio'] = True
            elif isinstance(result, str):
                outputs['text'] += "\n" + result
                outputs['show_text'] = True

        if not any(outputs.values()):
            outputs['text'] = "输出失败，请检查您输入的指令是否正确"
            outputs['show_text'] = True

        text_update = gr.update(visible=outputs['show_text'], value=outputs['text']) if outputs['show_text'] else gr.update(visible=False)
        image_update = gr.update(visible=outputs['show_image'], value=outputs['image']) if outputs['show_image'] else gr.update(visible=False)
        audio_update = gr.update(visible=outputs['show_audio'], value=outputs['audio']) if outputs['show_audio'] else gr.update(visible=False)
        video_update = gr.update(visible=outputs['show_video'], value=outputs['video']) if outputs['show_video'] else gr.update(visible=False)

        return text_update, image_update, audio_update, video_update
    except Exception as e:
        print(f"处理输入时发生错误: {e}")
        outputs = {
            'text': "输出失败，请检查您输入的指令是否正确",
            'show_text': True
        }
        return gr.update(visible=True, value=outputs['text']), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

with gr.Blocks(css="""
.gradio-container {background-color: #f5f5f5; text-align: center;}
.gr-button {width: 100%;}
""") as demo:
    gr.Markdown("<h1 style='text-align: center;'>WorkBrain V1.0 Demo</h1>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 用户指令")
            text_input = gr.Textbox(label="输入您的指令", placeholder="例如：帮我写一篇《春天来了》的文章，并为这篇文章配一张图片。", lines=4)
            file_input = gr.File(label="上传文件（支持图片、音频、视频及常见文档格式，可选）")

        with gr.Column():
            gr.Markdown("## 输出内容")
            with gr.Row():
                text_output = gr.Textbox(label="文本内容", lines=8, visible=True)
                image_output = gr.Image(label="图片内容", visible=False)
                audio_output = gr.Audio(label="音频内容", visible=False)
                video_output = gr.Video(label="视频内容", visible=False)

    submit_button = gr.Button("提交")
    
    submit_button.click(
        fn=process_input, 
        inputs=[text_input, file_input], 
        outputs=[text_output, image_output, audio_output, video_output]
    )

demo.launch(server_name="172.168.80.36", server_port=8006, share=True)