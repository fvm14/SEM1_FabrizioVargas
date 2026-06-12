import os
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATA_PROC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'processed'
)

def normalize_minmax(X_train, X_test, nombre_dataset):
    """
    Escala todas las features al rango [0, 1] usando Min-Max normalization.

    IMPORTANTE: el scaler aprende los parametros (min, max) SOLO del conjunto
    de entrenamiento. Luego aplica esos mismos parametros al test.
    Esto evita data leakage — el test no influye en el proceso de normalizacion.

    Referencia: Sangoleye et al. (2024)
    """
    os.makedirs(DATA_PROC, exist_ok=True)

    scaler = MinMaxScaler()

    X_train_normalizado = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns
    )

    X_test_normalizado = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )

    scaler_path = os.path.join(DATA_PROC, f'scaler_{nombre_dataset}.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    return X_train_normalizado, X_test_normalizado
