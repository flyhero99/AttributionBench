#!/bin/bash

while true; do
	rm -r /ML-A100/home/xiangyue/models/LongAlpaca-7B
    git clone https://huggingface.co/Yukang/LongAlpaca-7B && break
    sleep 5  # 可选：等待 5 秒后再次尝试
done
