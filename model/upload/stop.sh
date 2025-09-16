#!/bin/bash

# 停止脚本
# 使用方法: bash stop.sh

if [ -f "upload_pid.txt" ]; then
    PID=$(cat upload_pid.txt)
    echo "尝试停止进程 PID: $PID"
    
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID
        echo "✅ 进程已停止"
        rm upload_pid.txt
    else
        echo "进程已经不在运行"
        rm upload_pid.txt
    fi
else
    echo "没有找到 upload_pid.txt 文件"
    echo "没有后台进程在运行"
fi