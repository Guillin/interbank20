#!/bin/sh


# echo "******************* TRAINING CV MODELS *******************"

# FOLDS=3

# echo
# python train_cvmodel.py --model decision_tree_gini --metric roc_auc --folds $FOLDS --kind classification
# echo
# #python train_cvmodel.py --model decision_tree_entropy --metric roc_auc --folds $FOLDS --kind classification
# # echo
# python train_cvmodel.py --model rf --metric roc_auc --folds $FOLDS --kind classification
# echo
# python train_cvmodel.py --model lgbm --metric roc_auc --folds $FOLDS --kind classification
# echo
# python train_cvmodel.py --model xgb --metric roc_auc --folds $FOLDS --kind classification
# echo
# python train_cvmodel.py --model ctr --metric roc_auc --folds $FOLDS --kind classification
# echo
#echo "************************  END   ************************"


#echo "******************* TRAINING MODELS *******************"
#echo
# python train_model.py --model decision_tree_gini --metric roc_auc --kind classification
# echo
# python train_model.py --model decision_tree_entropy --metric roc_auc --kind classification
# echo
# python train_model.py --model rf --metric roc_auc --kind classification
# echo
#python train_model.py --model lgbm --metric roc_auc --kind classification
#echo
#python train_model.py --model xgb --metric roc_auc --kind classification
# echo
# python train_model.py --model ctr --metric roc_auc --kind classification
#echo
#echo "************************  END   ************************"

# echo "******************* PREDICT & SUBMIT TO KAGGLE  *******************"
# echo
# python predict_model.py --kind classification --model rf --input_file feateng_test_v6
# echo
# python predict_model.py --kind classification --model lgbm --input_file feateng_test_v7
# echo
# python predict_model.py --kind classification --model xgb --input_file feateng_test_v7
# echo
# python predict_model.py --kind classification --model ctr --input_file feateng_test_v7
# echo
# echo "************************  END   ************************"

