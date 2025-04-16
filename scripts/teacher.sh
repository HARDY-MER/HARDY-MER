# CMUMOSI
# 定义文件夹路径
dataset=$1
model=$2
folder_path="nohups_logs/teacher-${dataset}"

# 检查文件夹是否存在，如果不存在则创建
mkdir -p "$folder_path"

nohup bash scripts/teacher/a.sh 1 ${dataset} ${model} > ${folder_path}/a.log 2>&1 &
nohup bash scripts/teacher/t.sh 2 ${dataset} ${model} > ${folder_path}/t.log 2>&1 &
nohup bash scripts/teacher/v.sh 3 ${dataset} ${model} > ${folder_path}/v.log 2>&1 &
nohup bash scripts/teacher/at.sh 4 ${dataset} ${model} > ${folder_path}/at.log 2>&1 &
nohup bash scripts/teacher/av.sh 5 ${dataset} ${model} > ${folder_path}/av.log 2>&1 &
nohup bash scripts/teacher/tv.sh 6 ${dataset} ${model} > ${folder_path}/tv.log 2>&1 &