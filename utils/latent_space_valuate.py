from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils.feature_extraction import extract_embeddings

def evaluate_latent_space(model, dataloaders, device, class_names=None):
    """
    Valuta lo spazio latente del VAE usando un classificatore supervisionato.
    Args:
        model: il VAE già allenato
        dataloaders: dict con {"train": dl_train, "val": dl_val, "test": dl_test}
        device: 'cuda' 'mps' o 'cpu'
    """
    print("Estrazione embeddings...")
    z_train, y_train = extract_embeddings(model, dataloaders["train"], device, mode="z")
    z_test, y_test = extract_embeddings(model, dataloaders["test"], device, mode="z")

    print("Addestramento classificatore (SVM)...")
    clf = SVC(kernel='rbf', class_weight='balanced')  # kernel usato: gaussiano, oppure usare kernel='linear', 'poly', etc.
    clf.fit(z_train, y_train)

    print("Valutazione su test set...")
    y_pred = clf.predict(z_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("Confusion Matrix (Latent Space)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
