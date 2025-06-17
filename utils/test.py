import torch
from sklearn.metrics import classification_report

def test_classifier(model, data_loader, device, experiment=None, title="Model Evaluation"):
    # Imposta il modello in modalità valutazione (disabilita dropout/batchnorm)
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    # Disabilita il calcolo dei gradienti per velocizzare l'inferenza
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch)

            # Prende la classe con la probabilità più alta
            _, preds = torch.max(outputs, 1)

            # Conta il numero di predizioni corrette
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

            # Salva tutte le predizioni e le etichette reali per il report
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    # Calcola le metriche di classificazione
    report = classification_report(all_labels, all_preds, output_dict=True)
    acc = report["accuracy"]
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]
    
    # Logga le metriche
    if experiment:
        experiment.log_metric(f"{title}accuracy", acc)
        experiment.log_metrics({
            f"{title}precision": precision,
            f"{title}recall": recall,
            f"{title}f1_score": f1
        })

    # Stampa le metriche principali
    print(f"{title} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    
    # Restituisce le metriche e il report completo
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": report
    }
