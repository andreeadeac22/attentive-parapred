"""
Helper functions for plotting the ROC curve and the Precision-Recall curve.
"""
from sklearn import metrics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_roc_curve(labels_test, probs_test, colours=("#0072CF", "#68ACE5"),
                   label="This method", plot_fig=None):
    if plot_fig is None:
        plot_fig = plt.figure(figsize=(3.7, 3.7), dpi=400)
    ax = plot_fig.gca()

    num_runs = len(labels_test)
    tprs = np.zeros((num_runs, 10000))
    fprs = np.linspace(0.0, 1.0, num=10000)

    for i in range(num_runs):
        l = labels_test[i]
        p = probs_test[i]

        fpr, tpr, _ = metrics.roc_curve(l.flatten(), p.flatten())

        for j, fpr_val in enumerate(fprs):  # Inefficient, but good enough
            for t, f in zip(tpr, fpr):
                if f >= fpr_val:
                    tprs[i, j] = t
                    break

    avg_tpr = np.average(tprs, axis=0)
    err_tpr = np.std(tprs, axis=0)

    ax.plot(fprs, avg_tpr, c=colours[0], label=label)

    btm_err = avg_tpr - 2 * err_tpr
    btm_err[btm_err < 0.0] = 0.0
    top_err = avg_tpr + 2 * err_tpr
    top_err[top_err > 1.0] = 1.0

    ax.fill_between(fprs, btm_err, top_err, facecolor=colours[1])

    ax.set_ylabel("True positive rate")
    ax.set_xlabel("False positive rate")

    ax.legend()

    return plot_fig

def plot_pr_curve(labels_test, probs_test, colours=("#0072CF", "#68ACE5"),
                  label="This method", plot_fig=None):
    if plot_fig is None:
        plot_fig = plt.figure(figsize=(4.5, 3.5), dpi=300)
    ax = plot_fig.gca()

    num_runs = len(labels_test)
    precs = np.zeros((num_runs, 10000))
    recalls = np.linspace(0.0, 1.0, num=10000)

    for i in range(num_runs):
        l = labels_test[i]
        p = probs_test[i]

        #print("run i", i)
        #print("labels", l)
        #print("probs", p)

        prec, rec, _ = metrics.precision_recall_curve(l.flatten(), p.flatten())

        # Maximum interpolation
        for j in range(len(prec)):
            prec[j] = prec[:(j+1)].max()

        prec = list(reversed(prec))
        rec = list(reversed(rec))

        for j, recall in enumerate(recalls):  # Inefficient, but good enough
            for p, r in zip(prec, rec):
                if r >= recall:
                    precs[i, j] = p
                    break

    avg_prec = np.average(precs, axis=0)
    err_prec = np.std(precs, axis=0)

    ax.plot(recalls, avg_prec, c=colours[0], label=label)

    btm_err = avg_prec - 2 * err_prec
    btm_err[btm_err < 0.0] = 0.0
    top_err = avg_prec + 2 * err_prec
    top_err[top_err > 1.0] = 1.0

    ax.fill_between(recalls, btm_err, top_err, facecolor=colours[1])

    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")
    ax.legend()

    return plot_fig