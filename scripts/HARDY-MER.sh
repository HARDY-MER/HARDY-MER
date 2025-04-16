# IEMOCAPFour
# 定义文件夹路径
top_k=$1
dataset=$2
folder_path="nohups_logs/our-${dataset}"
student_model='OurModel'

# 检查文件夹是否存在，如果不存在则创建
mkdir -p "$folder_path"

nohup bash scripts/our/a.sh 0 ${top_k} ${dataset} ${student_model} > ${folder_path}/a-top_k_acc-K${top_k}-50.log 2>&1 &
nohup bash scripts/our/t.sh 1 ${top_k} ${dataset} ${student_model} > ${folder_path}/t-top_k_acc-K${top_k}-50.log 2>&1 &
nohup bash scripts/our/v.sh 2 ${top_k} ${dataset} ${student_model} > ${folder_path}/v-top_k_acc-K${top_k}-50.log 2>&1 &
nohup bash scripts/our/at.sh 5 ${top_k} ${dataset} ${student_model} > ${folder_path}/at-top_k_acc-K${top_k}-50.log 2>&1 &
nohup bash scripts/our/av.sh 5 ${top_k} ${dataset} ${student_model} > ${folder_path}/av-top_k_acc-K${top_k}-50.log 2>&1 &
nohup bash scripts/our/tv.sh 5 ${top_k} ${dataset} ${student_model} > ${folder_path}/tv-top_k_acc-K${top_k}-50.log 2>&1 &

# weighted: 特征加权
# V1：不带标签
# V2：带标签
# mean：距离平均
# TK1: 文本可用条件下，不检索特征
# TC: 对为本使用余弦相似度
# FT: 使用微调过的特征
# ab_dynamic: 消融动态检索