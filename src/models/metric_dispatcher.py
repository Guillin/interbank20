from sklearn import metrics
import numpy as np


def true_positive(y_true, y_pred): 
	"""
	Function to calculate True Positives 
    :param y_true: list of true values 
	:param y_pred: list of predicted values 
    :return: number of true positives
	"""
	# initialize
	tp = 0
	for yt, yp in zip(y_true, y_pred):
	    if yt == 1 and yp == 1: 
		    tp += 1
	return tp

def true_negative(y_true, y_pred): 
    """
    Function to calculate True Negatives 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: number of true negatives
    """
    # initialize
    tn = 0
    for yt, yp in zip(y_true, y_pred):
	    if yt == 0 and yp == 0: 
		    tn += 1
    return tn

def false_positive(y_true, y_pred): 
    """
    Function to calculate False Positives 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: number of false positives
    """
	# initialize
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1: 
            fp += 1
    return fp

def false_negative(y_true, y_pred): 
	"""
	Function to calculate False Negatives 
    :param y_true: list of true values 
    :param y_pred: list of predicted values 
    :return: number of false negatives
	"""
	# initialize
	fn = 0
	for yt, yp in zip(y_true, y_pred):
		if yt == 1 and yp == 0: 
			fn += 1
	return fn

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def mcc(y_true, y_pred):
    """
    This function calculates Matthew's Correlation Coefficient for binary classification.
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: mcc score
    """
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    numerator = (tp* tn) - (fp * fn)

    denominator = ((tp + fp)*(fn + tn)*(fp + tn)*(tp + fn))

    denominator = denominator ** 0.5

    return numerator/denominator



metrics_score = {
"roc_auc": metrics.roc_auc_score,
"accuracy": metrics.accuracy_score,
"mcc": mcc,
"apk": apk
}

