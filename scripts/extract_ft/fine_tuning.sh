# ${dataset}
export OMP_NUM_THREADS=3
gpu_id=$1
dataset=$2
model_name=$3
modality=$4
epoch=$5

python -u train_fine_tuning.py --dataset=${dataset} \
--audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT  --video-feature=manet_UTT \
--seed=152 --batch-size=16 --epoch=${epoch} --lr=0.0001 \
--model_name=${model_name} --gpu=${gpu_id} --modality=${modality}
