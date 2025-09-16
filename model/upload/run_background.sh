#!/bin/bash

# 后台运行脚本
# 使用方法: bash run_background.sh

echo "开始在后台运行 main.py..."

# 激活conda环境并在后台运行main.py
# nohup 确保即使终端关闭也继续运行
# 输出重定向到日志文件
nohup conda run -n qwen-agentic python main.py > upload_log.txt 2>&1 &

# 获取进程ID
PID=$!
echo "后台进程已启动，PID: $PID"
echo "日志文件: upload_log.txt"
echo ""
echo "监控命令:"
echo "  查看实时日志: tail -f upload_log.txt"
echo "  查看进程状态: ps aux | grep $PID"
echo "  杀死进程: kill $PID"
echo ""
echo "进程ID已保存到 upload_pid.txt"
echo $PID > upload_pid.txt