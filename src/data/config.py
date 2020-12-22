# config.py
INPUT_PATH = "../../data/raw/"
OUTPUT_PATH = "../../data/processed/"
NJOBS = -1
VERBOSE = 1
PLOT = False
SEED = 47
CSV_SEP = "," # specify type of separator used in csv in load method

# Specify numerical and categorical features in order to be processed adecually
num_features = ["age", "campaign", "pdays", "previous", "emp.var.rate", 
                "cons.price.idx", "cons.conf.idx","euribor3m", "nr.employed"]

cat_features = ["job", "marital", "education","default", "housing", "loan",
                "contact", "month", "day_of_week", "poutcome"]

# Specify which would be the target feature
target = "target"