from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io
from utils.feature_extraction import extract_embeddings_latent_space
import comet_ml

def evaluate_latent_space(model, dataloaders, device, class_names=None, experiment=None):
    """
    Valuta lo spazio latente del VAE usando un classificatore SVM.
    
    Args:
        model: il VAE già allenato
        dataloaders: dict con {"train": dl_train, "val": dl_val, "test": dl_test}
        device: 'cuda' 'mps' o 'cpu'
        class_names: lista dei nomi delle classi
        experiment: oggetto comet_ml.Experiment
    """
    print("Estrazione embeddings...")
    z_train, y_train = extract_embeddings_latent_space(model, dataloaders["train"], device)
    z_test, y_test = extract_embeddings_latent_space(model, dataloaders["test"], device)

    print("Addestramento classificatore (SVM)...")
    clf = SVC(kernel='rbf', class_weight='balanced')
    clf.fit(z_train, y_train)

    print("Valutazione su test set...")
    y_pred = clf.predict(z_test)

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=False)
    print(report)

    if experiment:
        experiment.log_text(report, metadata={"type": "classification_report"})

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("Confusion Matrix (Latent Space)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    if experiment:
        buf = io.BytesIO()
        plt.savefig("./images/confusion_matrix_latent_space_evaluate.png", buf)
        buf.seek(0)
        experiment.log_image(buf, name="confusion_matrix_latent_space_evaluate", image_format="png", step=None)
        buf.close()