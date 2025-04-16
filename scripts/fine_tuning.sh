export OMP_NUM_THREADS=3
model_name=$1

# IEMOCAPFour
#nohup bash scripts/extract_ft/fine_tuning.sh 0 IEMOCAPFour FineTuning audio > nohups_logs/ FineTuning_IEMOCAPFour_audio.log 2>&1 &
#nohup bash scripts/extract_ft/fine_tuning.sh 1 IEMOCAPFour FineTuning video > nohups_logs/ FineTuning_IEMOCAPFour_video.log 2>&1 &
#nohup bash scripts/extract_ft/fine_tuning.sh 2 IEMOCAPFour FineTuning text > nohups_logs/ FineTuning_IEMOCAPFour_text.log 2>&1 &

# IEMOCAPSix
#nohup bash scripts/extract_ft/fine_tuning.sh 0 IEMOCAPSix FineTuning audio > nohups_logs/ FineTuning_IEMOCAPSix_audio.log 2>&1 &
#nohup bash scripts/extract_ft/fine_tuning.sh 1 IEMOCAPSix FineTuning video > nohups_logs/ FineTuning_IEMOCAPSix_video.log 2>&1 &
#nohup bash scripts/extract_ft/fine_tuning.sh 2 IEMOCAPSix FineTuning text > nohups_logs/ FineTuning_IEMOCAPSix_text.log 2>&1 &

# CMUMOSEI
nohup bash scripts/extract_ft/fine_tuning.sh 2 CMUMOSEI CMUAudioFineTuning audio 30 > nohups_logs/FineTuning_CMUMOSEI_audio.log 2>&1 &
#nohup bash scripts/extract_ft/fine_tuning.sh 3 CMUMOSEI CMUVideoFineTuning video 30 > nohups_logs/FineTuning_CMUMOSEI_video.log 2>&1 &
#nohup bash scripts/extract_ft/fine_tuning.sh 4 CMUMOSEI CMUTextFineTuning text 30 > nohups_logs/FineTuning_CMUMOSEI_text.log 2>&1 &
#试试用第二好的模型作为特征提取的模型