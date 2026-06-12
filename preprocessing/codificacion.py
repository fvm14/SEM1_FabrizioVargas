import pandas as pd

def encode_categorical(df, nombre_dataset):
    """
    Aplica One-Hot Encoding a las columnas categoricas (texto).
    Las redes neuronales no pueden procesar texto directamente.

    Solo NSL-KDD y UNSW-NB15 tienen columnas categoricas.
    CICIDS2017 ya es completamente numerico.
    """

    if nombre_dataset == 'nslkdd':
        columnas_categoricas = ['protocol_type', 'service', 'flag']
        df = pd.get_dummies(df, columns=columnas_categoricas, prefix=columnas_categoricas)

    elif nombre_dataset == 'unswnb15':
        columnas_categoricas = [c for c in ['proto', 'service', 'state'] if c in df.columns]
        for col in columnas_categoricas:
            df[col] = df[col].astype(str)
        df = pd.get_dummies(df, columns=columnas_categoricas)

    return df

def align_columns(df_train, df_test):
    """
    Despues del One-Hot Encoding, el test puede tener categorias
    que el train no vio (o viceversa). Esta funcion sincroniza las columnas:
    - Columnas que faltan en test -> se agregan con valor 0
    - Columnas extra en test que no estan en train -> se eliminan
    """

    columnas_faltantes = set(df_train.columns) - set(df_test.columns)
    columnas_extra     = set(df_test.columns)  - set(df_train.columns)

    for col in columnas_faltantes:
        df_test[col] = 0

    df_test.drop(columns=list(columnas_extra), inplace=True, errors='ignore')

    df_test = df_test[df_train.columns]

    return df_test
