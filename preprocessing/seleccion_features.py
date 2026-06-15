import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results'
)

def select_features_gini(X_train, y_train, nombre_dataset):
    os.makedirs(RESULTS, exist_ok=True)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importancias = rf.feature_importances_
    nombres_features = X_train.columns.tolist()

    indices_ordenados = np.argsort(importancias)[::-1]

    ranking_df = pd.DataFrame({
        'feature':    [nombres_features[i] for i in indices_ordenados],
        'importance': importancias[indices_ordenados]
    })
    ranking_path = os.path.join(RESULTS, f'feature_importance_{nombre_dataset}.csv')
    ranking_df.to_csv(ranking_path, index=False)

    importancia_acumulada = np.cumsum(importancias[indices_ordenados])
    num_features_seleccionadas = int(np.searchsorted(importancia_acumulada, 0.95)) + 1

    features_seleccionadas = [nombres_features[i] for i in indices_ordenados[:num_features_seleccionadas]]

    print(f"  [{nombre_dataset}] features: {len(nombres_features)} -> {len(features_seleccionadas)} (95% importancia Gini)")

    return features_seleccionadas
