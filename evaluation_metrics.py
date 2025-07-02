from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def evaluate_predictions(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    return {"Accuracy": acc, "AUC": auc, "F1": f1}
