from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
)

def measure_results(graph):
    # Extracting the 'pred' and 'label' attributes
    pred = [data['pred'] for _, data in graph.nodes(data=True)]
    label = [data['label'] for _, data in graph.nodes(data=True)]

    # Calculate accuracy
    accuracy = accuracy_score(label, pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(label, pred))

    # Confusion matrix to compute TPR and FPR
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    tpr = tp / (tp + fn)  # True Positive Rate (Recall)
    fpr = fp / (fp + tn)  # False Positive Rate
    
    print(f"True Positive Rate (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    
    # AUROC calculation
    # Ensure binary classification where label and pred are in {0, 1}
    if len(set(label)) == 2:  # Check if binary classification
        auroc = roc_auc_score(label, pred)
        print(f"Area Under the ROC Curve (AUROC): {auroc:.4f}")
        
        # Optional: ROC Curve points
        fpr_values, tpr_values, thresholds = roc_curve(label, pred)
        print("ROC Curve Points:")
        for i, (f, t) in enumerate(zip(fpr_values, tpr_values)):
            print(f"Threshold {thresholds[i]:.2f}: FPR = {f:.4f}, TPR = {t:.4f}")
    else:
        print("AUROC calculation is only valid for binary classification.")

