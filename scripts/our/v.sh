# ${dataset}
export OMP_NUM_THREADS=2
gpu_id=$1
k=$2
dataset=$3
teacher_model='EasyReconstructModel'
student_model=$4

python -u train_our.py --dataset=${dataset} \
--audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT \
--seed=152 --batch-size=16 --epoch=50 --lr=0.0001 --hidden=256 --depth=4 --num_heads=2 \
--drop_rate=0.5 --attn_drop_rate=0.0 --test_condition=v --stage_epoch=25 --gpu=${gpu_id} \
--teacher_model=${teacher_model} --student_model=${student_model} --top_k=${k}