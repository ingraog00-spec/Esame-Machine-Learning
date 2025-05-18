import comet_ml
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import os
from models.model_classifier import LabelSmoothingLoss

def train_classifier(model, train_loader, val_loader, config, device, experiment):
    # Configurazione
    cfg = config["train_classifier"]

    # Nomi delle labels
    class_names = config.get("class_names", [str(i) for i in range(7)])

    # Inizializzazione della funzione di loss
    criterion = LabelSmoothingLoss(smoothing=0.2)

     # Inizializzazione ottimizzatore Adam
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay", 0))

    # Scheduler per il learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.05, patience=5)

    model.to(device)

    # Inizializzazione di metriche e variabili per early stopping
    best_acc = 0
    patience = 10
    patience_counter = 0
    train_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1s = []

    # Loop principale di addestramento
    for epoch in range(cfg["epochs"]):
        model.train()  # Imposta il modello in modalità training
        running_loss = 0.0

        # Barra di avanzamento per il caricamento dei batch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}", leave=False)

        # Iterazione su tutti i batch dell'epoca
        for x_batch, y_batch in progress_bar:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Azzera i gradienti
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x_batch)

            # Calcolo della loss
            loss = criterion(outputs, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulo della loss
            running_loss += loss.item()

        # Calcolo della media della loss per epoca
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        experiment.log_metric(f"classifier_train_loss", avg_loss, step=epoch + 1)

        # ----------- VALIDAZIONE -----------
        model.eval()  # Modalità evaluation: disabilita dropout/batchnorm

        # Metriche per valutazione
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x_val, y_val in val_loader: #  Itera su tutti i batch
                x_val, y_val = x_val.to(device), y_val.to(device)

                # Inferenza del modello: ottiene le probabilità/logit per ciascuna classe
                outputs = model(x_val)

                # Estrae la classe con probabilità massima per ogni campione del batch
                _, preds = torch.max(outputs, 1)

                # Aggiorna il conteggio totale dei campioni correttamente classificati
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)

                all_preds.extend(preds.cpu().numpy())   # Predizioni del modello
                all_labels.extend(y_val.cpu().numpy())  # Etichette reali

        # Calcolo delle metriche sul validation set
        acc = correct / total
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        # Salvataggio delle metriche
        val_accuracies.append(acc)
        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1s.append(f1)

        # Logging delle metriche di validazione su Comet
        experiment.log_metrics({
            f"classifier_val_accuracy": acc,
            f"classifier_val_precision": precision,
            f"classifier_val_recall": recall,
            f"classifier_val_f1_score": f1
        }, step=epoch + 1)

        # Aggiornamento dello scheduler del learning rate
        scheduler.step(acc)

        print(f"[classifier] Epoch [{epoch+1}/{cfg['epochs']}] - Loss: {avg_loss:.4f} - Acc: {acc:.4f} - Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f}")

        # ----------- EARLY STOPPING e SALVATAGGIO MODELLO MIGLIORE -----------
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0

            save_path = cfg["save_path"]
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"[classifier] New best model saved with acc: {acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[classifier] Early stopping triggered after {epoch+1} epochs.")
                break

    # ----------- CLASSIFICATION REPORT E METRICHE FINALI -----------
    report = classification_report(all_labels, all_preds, output_dict=True)
    experiment.log_metrics({
        f"classifier_final_val_accuracy": report["accuracy"],
        f"classifier_final_val_precision": report["weighted avg"]["precision"],
        f"classifier_final_val_recall": report["weighted avg"]["recall"],
        f"classifier_final_val_f1_score": report["weighted avg"]["f1-score"]
    })

    # ----------- MATRICE DI CONFUSIONE -----------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Matrice di Confusione - classifier")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = f"./reconstructions/confusion_matrix_classifier.png"
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path)
    experiment.log_image(cm_path)
    plt.close()

    # ----------- CURVA LOSS TRAINING -----------
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, marker='o')
    plt.title(f"classifier - Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    loss_curve_path = f"./reconstructions/train_loss_curve_classifier.png"
    plt.savefig(loss_curve_path)
    experiment.log_image(loss_curve_path)
    plt.close()

    # Dizionario delle metriche finali
    return {
        "accuracy": report["accuracy"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    }