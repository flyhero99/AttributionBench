#!/usr/bin/env bash
set -x
set -e


# work on OSC
# use torchrun
# python run_generation.py --task src/configs/task_configs/test_task.yaml --agent src/configs/agent_configs/llama2-7b.yaml
# python run_generation.py --task src/configs/task_configs/test_task.yaml --agent src/configs/agent_configs/llama2-7b-chat.yaml
# python run_generation.py --task src/configs/task_configs/test_task.yaml --agent src/configs/agent_configs/llama2-13b.yaml
# python run_generation.py --task src/configs/task_configs/test_task.yaml --agent src/configs/agent_configs/llama2-13b-chat.yaml
# python run_generation.py --task src/configs/task_configs/test_task.yaml --agent src/configs/agent_configs/llama2-70b.yaml
# python run_generation.py --task src/configs/task_configs/test_task.yaml --agent src/configs/agent_configs/llama2-70b-chat.yaml


python run_generation.py --task src/configs/task_configs/test_task.yaml --agent src/configs/agent_configs/gpt-4.yaml
python run_generation.py --task src/configs/task_configs/test_task.yaml --agent src/configs/agent_configs/gpt-3.5-turbo.yaml
python run_generation.py --task src/configs/task_configs/test_task.yaml --agent src/configs/agent_configs/vicuna-7b-v1.3.yaml
python run_generation.py --task src/configs/task_configs/test_task.yaml --agent src/configs/agent_configs/flan-t5-xl.yaml
python run_generation.py --task src/configs/task_configs/test_task.yaml --agent src/configs/agent_configs/openchat_v3.2_super.yaml
python run_generation.py --task src/configs/task_configs/test_task.yaml --agent src/configs/agent_configs/OpenOrca-Platypus2-13B.yaml
python run_generation.py --task src/configs/task_configs/test_task.yaml --agent src/configs/agent_configs/tulu-7b.yaml
python run_generation.py --task src/configs/task_configs/test_task.yaml --agent src/configs/agent_configs/wizard-13b-v1.2.yaml

