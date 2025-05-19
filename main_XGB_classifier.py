import comet_ml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from comet_ml import Experiment

def print_step(msg):
    print(f"\n{msg}\n{'='*60}")

def main():
    # Inizializza un nuovo esperimento Comet
    experiment = Experiment()
    experiment.set_name("Simulazione XGBoost Skin Lesion Classification - GridSearchCV")
    start_time = time.time()

    print_step("1. Caricamento dei dati")
    # Carica i dati di training e test da file CSV
    df_train = pd.read_csv("features_skin_lesion.csv")
    df_test = pd.read_csv("features_skin_lesion_test.csv")

    print(f"Train samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")
    print("Esempio dati:")
    print(df_train.head())
    print(df_test.head())

    # Estrae le feature e le etichette dai dataframe
    X_train = df_train.drop(columns=["filename", "label"]).values
    y_train_raw = df_train["label"].values

    X_test = df_test.drop(columns=["filename", "label"]).values
    y_test_raw = df_test["label"].values

    # Codifica le etichette in valori numerici
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    print_step("2. Inizio GridSearchCV con XGBoost")
    # Definisce la griglia di iperparametri per la ricerca
    param_grid = {
        'n_estimators': [100, 200],    # Numero di alberi
        'max_depth': [4, 6],           # Profondità massima di ciascun albero
        'learning_rate': [0.01, 0.05], # Tasso di apprendimento
        'subsample': [0.8, 1.0],       # Percentuale di campioni usati per costruire ciascun albero
        'colsample_bytree': [0.8, 1.0] # Percentuale di feature usate per costruire ciascun albero
    }
    experiment.log_parameters(param_grid)

    # Inizializza il classificatore XGBoost
    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )

    # Imposta la ricerca a griglia con cross-validation
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=3,
        n_jobs=-1,
        verbose=3
    )

    # Esegue la ricerca degli iperparametri migliori
    grid_search.fit(X_train, y_train)
    experiment.log_metric("best_score", grid_search.best_score_)
    experiment.log_parameters(grid_search.best_params_)

    print_step("3. Risultati GridSearch")
    print(f"Best Score: {grid_search.best_score_:.4f}")
    print(f"Best Params: {grid_search.best_params_}")

    # Recupera il modello migliore trovato dalla ricerca
    best_model = grid_search.best_estimator_

    print_step("4. Valutazione sul test set")
    # Effettua le predizioni sul set di test
    y_pred = best_model.predict(X_test)

    # Stampa il report di classificazione
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Calcola le metriche di valutazione
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Logga le metriche su Comet ML
    experiment.log_metrics({
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1
    })

    print_step("5. Creazione Confusion Matrix")
    # Crea e salva la matrice di confusione
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("./images/confusion_matrix.png")
    plt.show()
    plt.close()

    print_step("6. Feature Importances")
    # Visualizza e salva l'importanza delle feature
    importance = best_model.feature_importances_
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(importance)), importance)
    plt.title("Feature Importances")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("./images/feature_importance.png")
    plt.show()
    plt.close()

    print_step("7. Distribuzione delle Classi")
    # Visualizza e salva la distribuzione delle classi nel training set
    plt.figure(figsize=(7, 4))
    sns.countplot(x=y_train, order=pd.Series(y_train).value_counts().index)
    plt.title("Distribuzione delle Classi")
    plt.xlabel("Label")
    plt.tight_layout()
    plt.savefig("./images/label_distribution.png")
    plt.show()
    plt.close()

    print_step("8. Salvataggio del modello")
    # Salva il modello migliore su disco
    joblib.dump(best_model, "best_xgb_model.joblib")
    print("Modello salvato in: best_xgb_model.joblib")

    # Calcola e stampa la durata totale del processo
    duration = time.time() - start_time
    print_step("FINE PROCESSO")
    print(f"Durata totale: {duration:.2f} secondi")

    # Logga immagini e modello su Comet ML
    experiment.log_image("./images/confusion_matrix.png", name="Confusion Matrix")
    experiment.log_image("./images/feature_importance.png", name="Feature Importance")
    experiment.log_image("./images/label_distribution.png", name="Label Distribution")
    experiment.log_model("best_xgb_model", "best_xgb_model.joblib")
    experiment.end()

if __name__ == "__main__":
    main()
