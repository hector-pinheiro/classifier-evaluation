import math

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def plot_features_importances(model, max_features=20, dpi=200):
    lgb.plot_importance(model.model, max_num_features=max_features, dpi=dpi)
    plt.show()

    lgb.plot_importance(model.model,
                        max_num_features=max_features,
                        title='Feature importance - gain',
                        importance_type='gain',
                        dpi=dpi)
    plt.show()


def plot_features_importances(model, max_features=20, dpi=200):
    lgb.plot_importance(model.model, max_num_features=max_features, dpi=dpi)
    plt.show()

    lgb.plot_importance(model.model,
                        max_num_features=max_features,
                        title='Feature importance - gain',
                        importance_type='gain',
                        dpi=dpi)
    plt.show()


def __distance(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


def __interval(x, y):
    return x >= 0 and y >= 0 and x <= 1 and y <= 1


def __plot_base(title, xlabel, ylabel):
    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams["figure.dpi"] = 200
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.yticks(np.arange(0.0, 1.05, 0.1))
    plt.grid(which="both", axis="both", c="lightgray", ls="--", lw=0.5)


def __roc_curve(ground_truth, scores, weights):
    roc = np.transpose(metrics.roc_curve(ground_truth, scores, sample_weight=weights))

    roc_aux = sorted(
        [(th, fp, tp) for fp, tp, th in roc if __interval(fp, tp)])

    fpr = np.array([fp for th, fp, _ in roc_aux])
    tpr = np.array([tp for th, _, tp in roc_aux])
    ths = np.array([th for th, _, _ in roc_aux])

    roc_auc = metrics.auc(fpr, tpr)

    aux_zip = zip(tpr, fpr, ths)
    roc_dist, roc_th, roc_x, roc_y = min(
        [(__distance(1, 0, tp, fp), th, fp, tp) for tp, fp, th in aux_zip])

    return fpr, tpr, ths, {
        "roc_auc": roc_auc,
        "roc_dist": roc_dist,
        "roc_th": roc_th,
        "roc_fpr": roc_x,
        "roc_tpr": roc_y}

def __get_value_if_list(data_list, index):
    if isinstance(data_list, list):
        value = data_list[index]
    else:
        value = data_list
    return value

def plot_roc_cuves(model_list,
                   data_list,
                   data_tag_list,
                   label_columns_list='label',
                   weight_columns_list=None,
                   curve_type='pos'):

    if curve_type == 'pos':
        __plot_base(
            "Receiver Operating Characteristic (ROC) Curve",
            "False Positive Rate (FPR or Fallout)",
            "True Positive Rate (TPR or Recall)")
    elif curve_type == 'neg':
        __plot_base(
            "Negative - Receiver Operating Characteristic (ROC) Curve",
            "False Negative Rate (FNR or Miss Rate)",
            "True Negative Rate (TNR or Specificity)")
    else:
        raise NameError("Invalid evaluation curve type (must be 'pos' or 'neg')")

    min_point = None
    for data_index, data in enumerate(data_list):

        data_tag = data_tag_list[data_index]

        label_column = __get_value_if_list(label_columns_list, data_index)
        labels = data[label_column].to_numpy()

        if weight_columns_list is not None:
            weight_column = __get_value_if_list(weight_columns_list, data_index)
            weights = data[weight_column].to_numpy()
        else:
            weights = np.ones(labels.shape, dtype='float32')

        for model in model_list:
            scores = model.prediction(data)

            fpr, tpr, roc_ths, roc_metrics = __roc_curve(labels, scores, weights)
            auc = round(roc_metrics["roc_auc"], 4)
            dist = round(roc_metrics["roc_dist"], 4)
            th = round(roc_metrics["roc_th"], 4)

            plt.xscale("log")

            label = f"{model.get_tag()}-{data_tag} (AUC: {auc}, Dist: {dist}, Th: {th})"

            if curve_type == 'pos':
                plt.plot(fpr, tpr, label=label)
                if min_point is None:
                    min_point = min(fpr)
                else:
                    min_point = min(fpr + [min_point])
            else:
                fnr = 1.0 - tpr
                tnr = 1.0 - fpr
                plt.plot(fnr, tnr, label=label)
                if min_point is None:
                    min_point = min(fnr)
                else:
                    min_point = min(fnr + [min_point])

            plt.plot(
                [0.0001, roc_metrics["roc_fpr"]],
                [1, roc_metrics["roc_tpr"]],
                ls="--")

    rc = np.arange(min_point, 1.0005, 0.001)
    plt.plot(rc, rc, label="Random Classifier", c="darkgray", ls="--")

    plt.legend()
    # plt.savefig(output_name, dpi=200)
    # plt.clf()
    plt.show()
