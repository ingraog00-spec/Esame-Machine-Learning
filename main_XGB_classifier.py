import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

def print_step(msg):
    print(f"\n{msg}\n{'='*60}")

def main():
    start_time = time.time()

    print_step("1. Caricamento dei dati")
    df = pd.read_csv("features_skin_lesion.csv")
    print(f"Totale campioni: {len(df)}")
    print("Esempio dati:")
    print(df.head())

    X = df.drop(columns=["filename", "label"]).values
    y = df["label"].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print_step("2. Suddivisione Train/Test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Train size: {len(X_train)} - Test size: {len(X_test)}")

    print_step("3. Inizio GridSearchCV con XGBoost")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb = XGBClassifier(
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=3,
        n_jobs=-1,
        verbose=3
    )

    grid_search.fit(X_train, y_train)

    print_step("4. Risultati GridSearch")
    print(f"Best Score: {grid_search.best_score_:.4f}")
    print(f"Best Params: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_

    print_step("5. Valutazione sul test set")
    y_pred = best_model.predict(X_test)

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print_step("6. Creazione Confusion Matrix")
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

    print_step("7. Feature Importances")
    importance = best_model.feature_importances_
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(importance)), importance)
    plt.title("Feature Importances")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("./images/feature_importance.png")
    plt.show()

    print_step("8. Distribuzione delle Classi")
    plt.figure(figsize=(7, 4))
    sns.countplot(x=y, order=pd.Series(y).value_counts().index)
    plt.title("Distribuzione delle Classi")
    plt.xlabel("Label (codificata)")
    plt.tight_layout()
    plt.savefig("./images/label_distribution.png")
    plt.show()

    print_step("9. Salvataggio del modello")
    joblib.dump(best_model, "best_xgb_model.joblib")
    print("Modello salvato in: best_xgb_model.joblib")

    duration = time.time() - start_time
    print_step("FINE PROCESSO")
    print(f"Durata totale: {duration:.2f} secondi")

if __name__ == "__main__":
    main()
