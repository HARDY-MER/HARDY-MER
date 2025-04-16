export OMP_NUM_THREADS=3
#model_name=$1

# IEMOCAPFour
#nohup bash scripts/extract_ft/extract_ft.sh 0 IEMOCAPFour FineTuning audio > nohups_logs/extract_ft_IEMOCAPFour_audio.log 2>&1 &
#nohup bash scripts/extract_ft/extract_ft.sh 1 IEMOCAPFour FineTuning video > nohups_logs/extract_ft_IEMOCAPFour_video.log 2>&1 &
#nohup bash scripts/extract_ft/extract_ft.sh 2 IEMOCAPFour FineTuning text > nohups_logs/extract_ft_IEMOCAPFour_text.log 2>&1 &

# IEMOCAPSix
#nohup bash scripts/extract_ft/extract_ft.sh 0 IEMOCAPSix FineTuning audio> nohups_logs/extract_ft_IEMOCAPSix_audio.log 2>&1 &
#nohup bash scripts/extract_ft/extract_ft.sh 1 IEMOCAPSix FineTuning video> nohups_logs/extract_ft_IEMOCAPSix_video.log 2>&1 &
#nohup bash scripts/extract_ft/extract_ft.sh 2 IEMOCAPSix FineTuning text> nohups_logs/extract_ft_IEMOCAPSix_text.log 2>&1 &

# CMUMOSEI
nohup bash scripts/extract_ft/extract_ft.sh 2 CMUMOSEI CMUFineTuning audio > nohups_logs/extract_ft_CMUMOSEI_audio.log 2>&1 &
#nohup bash scripts/extract_ft/extract_ft.sh 3 CMUMOSEI CMUVideoFineTuning video > nohups_logs/extract_ft_CMUMOSEI_video.log 2>&1 &
#nohup bash scripts/extract_ft/extract_ft.sh 4 CMUMOSEI CMUTextFineTuning text > nohups_logs/extract_ft_CMUMOSEI_text.log 2>&1 &
