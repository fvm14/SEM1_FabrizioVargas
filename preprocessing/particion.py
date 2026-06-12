from sklearn.model_selection import train_test_split

def stratified_split(X, y, nombre_dataset):
    """
    Divide el dataset en 80% entrenamiento y 20% prueba.
    La division es ESTRATIFICADA: garantiza que la proporcion de clases
    (normal vs ataque) sea la misma en ambos subconjuntos.

    Solo se usa para CICIDS2017 porque NSL-KDD y UNSW-NB15
    ya vienen con su propia division oficial train/test.

    Referencia: Porrua et al. (2025)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    ratio_ataques_train = y_train.mean()
    ratio_ataques_test  = y_test.mean()

    print(f"  [{nombre_dataset}] train: {len(X_train)} filas | "
          f"ataques: {ratio_ataques_train:.4f}")
    print(f"  [{nombre_dataset}] test:  {len(X_test)} filas | "
          f"ataques: {ratio_ataques_test:.4f}")

    return X_train, X_test, y_train, y_test
