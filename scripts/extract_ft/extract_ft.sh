# ${dataset}
export OMP_NUM_THREADS=3
gpu_id=$1
dataset=$2
model_name=$3
modality=$4

python -u extract_ft_feat.py --dataset=${dataset} \
--audio-feature=wav2vec-large-c-UTT --text-feature=deberta-large-4-UTT --video-feature=manet_UTT \
--batch-size=16 --model_name=${model_name} --gpu=${gpu_id} --modality=${modality}
