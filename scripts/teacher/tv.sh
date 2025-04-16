# ${dataset}
export OMP_NUM_THREADS=3
gpu_id=$1
dataset=$2
model=$3

python -u train_teacher.py --dataset=${dataset} \
--audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT \
--seed=334 --batch-size=16 --epoch=200 --lr=0.0001 --hidden=256 --depth=4 --num_heads=2 \
--drop_rate=0.5 --attn_drop_rate=0.0 --test_condition=tv --stage_epoch=100 --gpu=${gpu_id} --model=${model}
