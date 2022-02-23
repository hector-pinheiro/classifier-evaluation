import numpy as np


def blocked_rate(scores, weights, threshold):
    return np.sum(weights[scores >= threshold]) / np.sum(weights)


def tpr(scores, labels, weights, threshold):
    pos_samples = labels == 1.0
    scores = scores[pos_samples]
    weights = weights[pos_samples]
    return np.sum(weights[scores >= threshold]) / np.sum(weights)


def fpr(scores, labels, weights, threshold):
    neg_samples = labels == 0.0
    scores = scores[neg_samples]
    weights = weights[neg_samples]
    return np.sum(weights[scores >= threshold]) / np.sum(weights)


def precision(scores, labels, weights, threshold):
    pos_decisions = scores >= threshold
    labels = labels[pos_decisions]
    weights = weights[pos_decisions]
    return np.sum(weights[labels == 1.0]) / np.sum(weights)


def approved_rate(scores, weights, threshold):
    return np.sum(weights[scores <= threshold]) / np.sum(weights)


def tnr(scores, labels, weights, threshold):
    return 1.0 - fpr(scores, labels, weights, threshold)


def fnr(scores, labels, weights, threshold):
    return 1.0 - tpr(scores, labels, weights, threshold)


def neg_precision(scores, labels, weights, threshold):
    neg_decicions = scores <= threshold
    labels = labels[neg_decicions]
    weights = weights[neg_decicions]
    return np.sum(weights[labels == 0.0]) / np.sum(weights)
