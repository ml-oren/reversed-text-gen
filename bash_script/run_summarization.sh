#!/bin/bash
#SBATCH --mail-user=matanel.oren@mail.huji.ac.il
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:1,vmem:20g
#SBATCH --mem=40g
#SBATCH --time=1-0
#SBATCH -c4
#SBATCH --job-name=bart
#SBATCH --output=/cs/labs/roys/matanel.oren/ANLP/project/logs/sbatch_%J.out

cd /cs/labs/roys/matanel.oren/ANLP/project

module load cuda
source /cs/labs/roys/matanel.oren/venvs/lab_proj/bin/activate

# model=/cs/labs/roys/matanel.oren/ANLP/project/results_models/bart_from_pretrained_roberta_cnn/checkpoint-107670
# model=facebook/bart-large-cnn
# model=/cs/labs/roys/matanel.oren/ANLP/project/bert2bert/checkpoint-53835
model=/cs/labs/roys/matanel.oren/ANLP/project/models/pretrained_bart_from_roberta

tokenizer=roberta-base
# tokenizer=facebook/bart-large
# tokenizer=/cs/labs/roys/matanel.oren/ANLP/project/bert2bert/checkpoint-53835

WANDB_PROJECT=reversed-order-generation

echo "Starting evaluating ${model} on CNN/DailyMail"
python /cs/labs/roys/matanel.oren/ANLP/project/run_summarization.py \
    --model_name_or_path ${model} \
    --tokenizer_name ${tokenizer} \
    --dataset_name cnn_dailymail \
    --dataset_config_name "3.0.0" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --max_source_length 512 \
    --report_to wandb \
    --logging_dir /cs/labs/roys/matanel.oren/ANLP/project/logs \
    --logging_steps 200 \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --save_strategy epoch \
    --overwrite_output_dir \
    --predict_with_generate \
    --dataloader_num_workers 4 \
    --run_name finetuning-bart-from-roberta \
    --output_dir /cs/labs/roys/matanel.oren/ANLP/project/results_models/pretrained_bart_from_roberta \