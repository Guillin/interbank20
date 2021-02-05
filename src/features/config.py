# config.py
INPUT_PATH = "../../data/raw/"
OUTPUT_PATH = "../../data/features/"
NJOBS = -1
VERBOSE = 1
PLOT = False
SEED = 47
CSV_SEP = "," # specify type of separator used in csv in load method

# this option only available in gcp
USE_GCP = True
GCP_BUCKET_NAME = "interbank2020"
GCP_BUCKET_FOLDER_NAME = "features/"