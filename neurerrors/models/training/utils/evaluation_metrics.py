import numpy as np

def filter_outliers_iqr(coords):
    Q1 = np.percentile(coords, 25, axis=0)
    Q3 = np.percentile(coords, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 5 * IQR
    upper_bound = Q3 + 5 * IQR
    indices = np.where((coords >= lower_bound).all(axis=1) & (coords <= upper_bound).all(axis=1))
    return coords[indices], indices[0]  # Return filtered coords and indices

def confusion_matrix(true, pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, pred):
        #print(t,p)
        t = int(t)  # Ensure t is an integer
        p = int(p)  # Ensure p is an integer
        matrix[t, p] += 1
    return matrix

def calculate_metrics(conf_matrix):
    precision = []
    recall = []
    f1_scores = []
    for i in range(len(conf_matrix)):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        precision.append(tp / (tp + fp) if tp + fp > 0 else 0)
        recall.append(tp / (tp + fn) if tp + fn > 0 else 0)
        f1_scores.append(2 * precision[-1] * recall[-1] / (precision[-1] + recall[-1]) if precision[-1] + recall[-1] > 0 else 0)
    return precision, recall, f1_scores