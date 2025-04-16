# *_*coding:utf-8 *_*
import os
import sys
import socket

############ For LINUX ##############
# path
DATA_DIR = {
	'CMUMOSI': '/sdc/home/zuohaolin/zuohaolin/CL-MMIN/data/CMUMOSI',   # for nlpr
	'CMUMOSEI': '/sdc/home/zuohaolin/zuohaolin/CL-MMIN/data/CMUMOSEI',# for nlpr
	'IEMOCAPSix': '/sdc/home/zuohaolin/zuohaolin/CL-MMIN/data/IEMOCAP', # for nlpr
	'IEMOCAPFour': '/sdc/home/zuohaolin/zuohaolin/CL-MMIN/data/IEMOCAP', # for nlpr
}
PATH_TO_LABEL = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'CMUMOSI_features_raw_2way.pkl'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'CMUMOSEI_features_raw_2way.pkl'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'IEMOCAP_features_raw_6way.pkl'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'IEMOCAP_features_raw_4way.pkl'),
}
PATH_TO_FT_FEATURES = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'ft_features'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'ft_features'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'ft_features_six'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'ft_features_four'),
}
PATH_TO_FT_INDEX = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'ft_faiss_index'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'ft_faiss_index'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'ft_faiss_index_6'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'ft_faiss_index_4'),
}
PATH_TO_FEATURES = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'features'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'features'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'features'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'features'),
}
PATH_TO_INDEX = {
	'CMUMOSI': os.path.join(DATA_DIR['CMUMOSI'], 'faiss_index'),
	'CMUMOSEI': os.path.join(DATA_DIR['CMUMOSEI'], 'faiss_index'),
	'IEMOCAPSix': os.path.join(DATA_DIR['IEMOCAPSix'], 'faiss_index_6'),
	'IEMOCAPFour': os.path.join(DATA_DIR['IEMOCAPFour'], 'faiss_index_4'),
}

# dir
SAVED_ROOT = os.path.join('../saved')
DATA_DIR = os.path.join(SAVED_ROOT, 'data')
MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
LOG_DIR = os.path.join(SAVED_ROOT, 'log')
LOSS_DIR = os.path.join(SAVED_ROOT, 'loss')


