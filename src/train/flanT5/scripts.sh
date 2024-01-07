#!/bin/bash
#SBATCH --account=PAA0201
#SBATCH --job-name=flan-t5-xxl-combined
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=524288MB
#SBATCH --mail-type=BEGIN,END,FAIL
scontrol show job $SLURM_JOBID
qstat -f $SLURM_JOB_ID

module load python
module load miniconda3

cd $SLURM_SUBMIT_DIR

source activate llama
torchrun --nproc_per_node=4 --master_port=10086 train.py \
    --model_name_or_path google/flan-t5-xxl \
    --data_path ./final_data_504/train-combined.json\
    --output_dir ./flan-t5-xxl-combined \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --logging_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --tf32 True \
    --deepspeed ds_config_zero2.json
    

#!/bin/bash
#SBATCH --account=PAA0201
#SBATCH --job-name=flan-t5-xl-combined
#SBATCH --time=5:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mail-type=BEGIN,END,FAIL
scontrol show job $SLURM_JOBID
qstat -f $SLURM_JOB_ID

module load python
module load miniconda3

cd $SLURM_SUBMIT_DIR

source activate llama
torchrun --nproc_per_node=4 --master_port=10086 train.py \
    --model_name_or_path google/flan-t5-xl \
    --data_path ./final_data_504/train-combined.json\
    --output_dir ./flan-t5-xl-combined \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --logging_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --tf32 True \
    --deepspeed ds_config_zero2.json


#!/bin/bash
#SBATCH --account=PAA0201
#SBATCH --job-name=flan-t5-large-nli
#SBATCH --time=1:30:00
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
scontrol show job $SLURM_JOBID
qstat -f $SLURM_JOB_ID

module load python
module load miniconda3

cd $SLURM_SUBMIT_DIR

source activate llama
torchrun --nproc_per_node=2 --master_port=10086 train.py \
    --model_name_or_path google/flan-t5-large \
    --data_path ./final_data_504/train-nli.json\
    --output_dir ./flan-t5-large-nli \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --logging_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --bf16 True \
    --tf32 True
    
#!/bin/bash
#SBATCH --account=PAA0201
#SBATCH --job-name=flan-t5-large-nli
#SBATCH --time=0:15:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
scontrol show job $SLURM_JOBID
qstat -f $SLURM_JOB_ID

module load python
module load miniconda3

cd $SLURM_SUBMIT_DIR

source activate llama
python test.py