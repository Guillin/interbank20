# config.py
INPUT_PATH = "../../data/features/"
OUTPUT_PATH = "../../data/train_test/"
NJOBS = -1
VERBOSE = 2
PLOT = False
SEED = 47
TEST_SIZE = 0.3

# Specify numerical and categorical features in order to be processed adecually
num_features = ["age", "campaign", "pdays", "previous", "emp.var.rate", 
                "cons.price.idx", "cons.conf.idx","euribor3m", "nr.employed"]

cat_features = ["job", "marital", "education","default", "housing", "loan",
                "contact", "month", "day_of_week", "poutcome"]

# Specify which would be the target feature
target = "target"