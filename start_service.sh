#!/bin/bash

# 定义根目录
ROOT_DIR=$(pwd)

# 函数：启动服务并显示加载过程
start_service() {
  local dir=$1
  local script=$2
  local port=$3
  local log_file="${script%.*}.log" # 基于脚本名生成日志文件名

  echo "Starting service in $dir on port $port..."
  cd "$dir" || { echo "Directory not found: $dir"; exit 1; }

  # 使用tee命令将输出同时发送到文件和终端
  nohup python "$script" 2>&1 | tee -a "$log_file" &
  local pid=$! # 获取后台进程ID

  # 等待服务初始化完成或打印出关键信息
  sleep 5 # 可以根据实际情况调整等待时间
  echo "Service started with PID $pid. See logs for more details."

  # 返回到根目录
  cd "$ROOT_DIR"
}

# 启动门控网络服务
start_service "./model_gate_network" "predict_api.py" 8001

# 启动文本生成模型服务
start_service "./expert_models/text_generation" "infer_api.py" 8002

# 启动图像生成模型服务
start_service "./expert_models/image_generation" "infer_api.py" 8003

# 启动图像理解模型服务
start_service "./expert_models/image_understanding" "infer_api.py" 8004

# 所有服务已启动
echo "All services have been started."