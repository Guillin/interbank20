#!/bin/sh


# echo "******************* TRAINING CV MODELS *******************"

# FOLDS=3

# echo
# python train_cvmodel.py --model decision_tree_gini --metric roc_auc --folds $FOLDS --kind classification
# echo
# python train_cvmodel.py --model decision_tree_entropy --metric roc_auc --folds $FOLDS --kind classification
# echo
# python train_cvmodel.py --model rf --metric roc_auc --folds $FOLDS --kind classification
# echo
# python train_cvmodel.py --model lgbm --metric roc_auc --folds $FOLDS --kind classification
# echo
# python train_cvmodel.py --model xgb --metric roc_auc --folds $FOLDS --kind classification
# echo
# echo "************************  END   ************************"


echo "******************* TRAINING MODELS *******************"
echo
python train_model.py --model decision_tree_gini --metric roc_auc --kind classification
echo
python train_model.py --model decision_tree_entropy --metric roc_auc --kind classification
echo
python train_model.py --model rf --metric roc_auc --kind classification
echo
python train_model.py --model lgbm --metric roc_auc --kind classification
echo
python train_model.py --model xgb --metric roc_auc --kind classification
echo
echo "************************  END   ************************"