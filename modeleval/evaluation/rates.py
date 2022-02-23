import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from modeleval.evaluation.metrics import blocked_rate, tpr, fpr, precision, approved_rate, tnr, fnr, neg_precision


def data_description(data, label_column, weight_column=None):
    labels = data[label_column].to_numpy()
    pos_samples = data.iloc[labels == 1.0]
    neg_samples = data.iloc[labels == 0.0]

    n_samples = len(data)
    n_pos_samples = len(pos_samples)
    n_neg_samples = len(neg_samples)

    if weight_column is not None:
        samples_weights = data[weight_column].to_numpy()
        weights = np.sum(samples_weights)
        pos_weights = np.sum(samples_weights[labels == 1.0])
        neg_weights = np.sum(samples_weights[labels == 0.0])
    else:
        weights = float(n_samples)
        pos_weights = float(n_pos_samples)
        neg_weights = float(n_neg_samples)
        weight_column = 'weights'

    metrics = [['samples', n_samples],
               ['pos-samples', n_pos_samples],
               ['neg-samples', n_neg_samples],
               [weight_column, weights],
               ['pos-' + weight_column, pos_weights],
               ['neg-' + weight_column, neg_weights]]

    return pd.DataFrame(metrics, columns=['metric', 'value'])


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
        raise NameError("Invalid evaluation mode (must be 'blocking' or 'approval')")

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
            model.get_tag(),
            data_tag
        )
        metrics.append(metrics_values)

    return pd.DataFrame(metrics, columns=metrics_names)
