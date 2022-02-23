import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from modeleval.evaluation.metrics import blocked_rate, tpr, fpr, precision, approved_rate, tnr, fnr, neg_precision


def __find_threshold_index(rates, thrs, target_rate, mode):
    valid_indxs = np.arange(0, rates.shape[0])[rates <= target_rate]
    valid_thrs = thrs[rates <= target_rate]
    if mode == 'max':
        best_ind = np.argmax(valid_thrs)
    else:
        best_ind = np.argmin(valid_thrs)
    return valid_indxs[best_ind]


def find_thresholds(model, data, label_column, target_values, value_type):
    scores = model.prediction(data)
    labels = data[label_column].to_numpy()
    fpr, tpr, thrs = roc_curve(labels, scores)
    if value_type == 'fpr':
        rates = fpr
        mode = 'min'
    elif value_type == 'fnr':
        rates = 1.0 - tpr
        mode = 'max'
    else:
        raise NameError('Invalid value type when finding thresholds')

    thresholds = []
    opt_rates = []
    for target_value in target_values:
        ind = __find_threshold_index(rates, thrs, target_value, mode)
        thresholds.append(thrs[ind])
        opt_rates.append(rates[ind])

    return thresholds, opt_rates


def model_evaluation(scores, labels, weights, threshold, mode, model_tag='', data_tag=''):
    if mode == 'blocking':
        metrics = ['model', 'data', 'threshold', 'blocked-rate', 'tpr/recall', 'fpr', 'precision']
        values = [
            model_tag,
            data_tag,
            threshold,
            blocked_rate(scores, weights, threshold),
            tpr(scores, labels, weights, threshold),
            fpr(scores, labels, weights, threshold),
            precision(scores, labels, weights, threshold)
        ]
    elif mode == 'approval':
        metrics = ['model', 'data', 'threshold', 'approved-rate', 'tnr/specificity', 'fnr', 'neg-precision']
        values = [
            model_tag,
            data_tag,
            threshold,
            approved_rate(scores, weights, threshold),
            tnr(scores, labels, weights, threshold),
            fnr(scores, labels, weights, threshold),
            neg_precision(scores, labels, weights, threshold)
        ]
    else:
        raise NameError('Invalid evaluation mode (must be blocking or approval)')

    return metrics, values


def compute_evaluation_rates(model, data, label_column, thresholds, mode='blocking', weight_column=None, data_tag=''):
    scores = model.prediction(data)
    labels = data[label_column].to_numpy()
    if weight_column is not None:
        weights = data[weight_column].to_numpy()
    else:
        weights = np.ones(scores.shape, dtype='float32')

    metrics = []
    for threshold in thresholds:
        metrics_names, metrics_values = model_evaluation(
            scores,
            labels,
            weights,
            threshold,
            mode,
            model.input_name,
            data_tag
        )
        metrics.append(metrics_values)

    return pd.DataFrame(metrics, columns=metrics_names)
