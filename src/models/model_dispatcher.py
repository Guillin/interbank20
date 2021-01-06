# model_dispatcher.py
from sklearn import ensemble 
from sklearn import tree
from xgboost.sklearn import XGBClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



models = {
"decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini", max_depth=5 ),

"decision_tree_entropy": tree.DecisionTreeClassifier( criterion="entropy"),

"rf": ensemble.RandomForestClassifier(n_estimators=1000, criterion="gini", max_depth=3),

"lgbm": LGBMClassifier(n_estimators=1000, min_child_samples=250, boosting_type="gbdt"),

"xgb": XGBClassifier(booster = 'gbtree', objective  = 'binary:logistic', n_estimators=800,  eval_metric = 'auc', max_depth=3)

}