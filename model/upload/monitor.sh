#!/bin/bash

# 监控脚本
# 使用方法: bash monitor.sh

if [ -f "upload_pid.txt" ]; then
    PID=$(cat upload_pid.txt)
    echo "检查进程 PID: $PID"
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ 进程正在运行"
        echo "CPU和内存使用情况:"
        ps -p $PID -o pid,ppid,cmd,%mem,%cpu
        echo ""
        echo "最新日志 (最后10行):"
        echo "------------------------"
        tail -n 10 upload_log.txt
        echo "------------------------"
        echo ""
        echo "实时监控日志: tail -f upload_log.txt"
    else
        echo "❌ 进程已结束"
        echo "检查日志文件 upload_log.txt 查看结果"
    fi
else
    echo "没有找到 upload_pid.txt 文件"
    echo "请先运行 bash run_background.sh"
fi