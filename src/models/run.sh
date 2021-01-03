#!/bin/sh

echo "******************* TRAINING MODELS *******************"
echo
python train_model.py --model decision_tree_gini --metric roc_auc --folds 5 --kind classification
echo
python train_model.py --model rf --metric roc_auc --folds 5 --kind classification
echo
python train_model.py --model rf --metric roc_auc --folds 5 --kind classification
echo
echo "************************  END   ************************"