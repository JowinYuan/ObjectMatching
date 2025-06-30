import os
from openai import OpenAI
from PIL import Image
import base64
import json
import sys
import time

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

#从环境变量中获取api_key并且创建客户端
api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    log.error("环境变量中不存在DASHSCOPE_API_KEY，请设置API Key。")
    sys.exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取dataset文件夹的路径
DATASET_DIR = os.path.join(BASE_DIR, "datasets")

def getImage2Base64(image_path, target_size = None):
    """读取本地图片文件并且缩放到目标大小，同时编码为Base64字符
    Args:
        image_path(str): 本地图片文件路径
        target_size(truple): 目标图片大小,如果只提供了一个，另一个按等比缩放
    Returns:
        str: 本地图片的Base64编码
        None: 发生错误 
    """
    try:
        with Image.open(image_path) as img_file:
            if target_size:
                target_width, target_height = target_size
                origin_width, origin_height = img_file.size
                
                #计算缩放后图片尺寸
                if target_height and target_width:
                   new_size = (target_width, target_height)
                elif target_width:
                    ratio = target_width / origin_width  
                    new_size = (target_width, int(origin_height * ratio))
                elif target_height:
                    ratio = target_height / origin_height
                    new_size = (int(origin_width * ratio), target_height)
                else:
                    new_size = img_file.size
                
                img_file.resize(new_size, Image.LANCZOS)
            
            from io import BytesIO
            buffer = BytesIO()
            img_format = img_file.format or 'JPG'
            img_file.save(buffer, format=img_format)
            data = buffer.getvalue()
            return base64.b64encode(data).decode('utf-8')
        
    except FileNotFoundError:
        log.error(f"图像文件不存在: {image_path}")
        raise
    except Exception as e:
        log.error(f"该路径图片文件编码失败: {image_path}: {e}")
        raise

def encodeImagesInFolder(folder_path, size = None):
    """
        从给定文件夹路径读取所有图片文件并且编码为Base64
        Args:
            folder_path(str): 给定的文件夹路径
            size(truple): 目标缩放大小  
        Returns:
            truple(str): 文件夹所有图片文件的base64编码
    """
    supported_exts = {'.png', '.jpg', '.jpeg'}
    results = []

    if not os.path.isdir(folder_path):
        log.error(f"未找到指定文件夹: {folder_path}")
        return results

    for fname in os.listdir(folder_path):
        _, ext = os.path.splitext(fname)
        if ext.lower() in supported_exts:
            # 取出所有需要格式的文件路径
            full_path = os.path.join(folder_path, fname)
            try:
                b64 = getImage2Base64(full_path, size)
                # results.append((fname, b64))
                results.append(b64)
            except Exception:
                continue
    return results

def getJsonData(parent_dir, data_folder_name):
    """提取json具体内容以及获取拼接目录名
    Args:
        parent_dir(str): 父目录
        dataFolder(str): 想要使用的数据集的子目录名
    Returns:
        str: 与json文件同级路径
        truple: json文件内容
    Forms:
        "id": json_data[i]['id'] 当前任务id
        "video_folder": json_data[i]['input']['video_folder'] 视频文件所在子目录
        "prompt": json_data[i]['input']['prompt'], 问题
        "answer": json_data[i]['output']['answer'] 答案
    """    
    if not os.path.isdir(parent_dir):
        log.error(f"未找到指定文件夹: {parent_dir}")
        return None, []

    data_folder_path = os.path.join(parent_dir, data_folder_name)
    json_file_path = os.path.join(data_folder_path, data_folder_name + ".json")
    
    if not os.path.isfile(json_file_path):
        log.error(f"未找到数据集json文件: {json_file_path}")
        return None, []
    with open(json_file_path, "r", encoding="utf-8") as f:
        try:
            json_data = json.load(f)
        except json.JSONDecodeError as e:
            log.error(f"JSON解码错误: {e}")
            return None, []
    return data_folder_path, json_data["data"]

def save_to_json(data, output_folder, file_name):
    """
    将数据保存为json文件到指定文件夹
    Args:
        data: 要保存的数据（如dict或list）
        output_folder(str): 输出文件夹名
        file_name: 文件名（如 result.json）
    """
    folder_path = os.path.join(BASE_DIR, output_folder)
    # 如果目标文件夹不存在则创建
    if not os.path.exists(folder_path):
        log.info("目标文件夹不存在，正在创建: %s", folder_path)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False: 保持中文字符不被转义
        # indent=4: 控制缩进数，美化输出格式
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"已保存到: {file_path}")

def client_request(data_folder_name, size=None, output_folder=None):
    """API发送请求并获取响应结果,同时存储在json文件中
    Args:
        data_folder_name(str): 数据集目录名
        size(truple): 目标缩放大小
    Returns:
        list: 返回API响应结果
    """
    # data_folder_path: D:\Code\Pytorch\ObjectMatching\datasets\ColorMatch
    data_folder_path, json_data = getJsonData(DATASET_DIR, data_folder_name)
    if not data_folder_path or not json_data:
        log.error(f"无法获取数据集: {data_folder_name}")
        return None
    n = len(json_data)
    log.info(f"正在处理数据集: {data_folder_name}，共计任务数: {n}")
    start_time = time.time()  # 记录数据集开始处理的时间

    results = [] # 存储结果列表    

    for i in range(n):
        # 每处理50个任务打印一次时间信息
        if i > 0 and (i+1) % 50 == 0:  
            elapsed_time = time.time() - start_time
            remaining_tasks = n - i
            estimated_time = (elapsed_time / i) * remaining_tasks
            log.info(f"已处理任务数: {i}, 已花费时间: {elapsed_time:.2f}s, 预计剩余时间: {estimated_time:.2f}s")

        task_id = json_data[i]['id'] # 当前任务id
        vedio_folder = json_data[i]['input']['video_folder'] # 视频文件所在子目录
        prompt = json_data[i]['input']['prompt'] # 问题
        answer = json_data[i]['output']['answer'] # 答案

        # 拼接视频文件所在子目录的路径
        video_folder_path = os.path.join(data_folder_path, vedio_folder)
        # 编码视频文件夹下的所有图片
        images_b64 = encodeImagesInFolder(video_folder_path, size)
        if not images_b64:
            log.error(f"未找到有效图片文件，任务ID: {task_id}, 视频文件夹路径: {video_folder_path}")
            continue
        # 构建请求体
        try:
            completion = client.chat.completions.create(
                model="qvq-plus",
                messages=[{"role": "user","content": [
                    # 传入图像列表时，用户消息中的"type"参数为"video"
                    {"type": "video","video": [
                        f"data:image/png;base64,{images_b64[0]}",
                        f"data:image/png;base64,{images_b64[1]}",
                        f"data:image/png;base64,{images_b64[2]}",
                        f"data:image/png;base64,{images_b64[3]}",]},
                    {"type": "text","text": prompt + "Just response with number."},
                ]}],
                stream=True,
                # 解除以下注释会在最后一个chunk返回Token使用量
                # stream_options={
                #     "include_usage": True
                # }
            )

            answer_content = ""
            is_answering = False #保证只接收回复内容
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # 只处理最终回复内容
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        is_answering = True
                    # print(delta.content, end='', flush=True) #控制模型输出回答到控制台
                    answer_content += delta.content

            model_answer = answer_content.strip() #strip()去除首尾空格
            current_status = "success"

        except Exception as e:
            logging.error(f"API调用发生错误 for ID {task_id}: {e}")
            model_answer = f"API调用错误: {e}"
            current_status = "error"

        log.info(f"任务ID: {task_id}, 标准答案: {answer}, 模型回答: {model_answer}, 状态: {current_status}")

        # 构建结果列表
        results.append({
            'id': task_id,
            'prompt': prompt,
            'standard_answer': answer,
            'model_answer': model_answer,
            'status': current_status,
        })
    log.info(f"数据集: {data_folder_name} 处理完成，共计任务数: {n}")
    # 保存结果到json
    if output_folder:
        save_to_json(results, output_folder, f"{data_folder_name}_results.json")
    else:
        save_to_json(results, "ansCollect", f"{data_folder_name}_results.json")


    


    
    

if __name__ == "__main__":

    # 所有datasets文件夹下的子文件夹名
    data_folder_names = ["ColorMatch", "LOGOMarkerMatch", "MotionMatch", "ObjectMarkerMatch",
                "PositionMatch", "RelationMatch", "ShapeMatch", "SizeMatch"]
    client_request(data_folder_names[4], size=(640, 480), output_folder="ansCollect")



