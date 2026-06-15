import os
import pandas as pd

from preprocessing.carga_datasets    import load_nslkdd, load_cicids2017, load_unswnb15
from preprocessing.limpieza          import clean_dataset, binary_label
from preprocessing.codificacion      import encode_categorical, align_columns
from preprocessing.seleccion_features import select_features_gini
from preprocessing.normalizacion     import normalize_minmax
from preprocessing.particion         import stratified_split

DATA_PROC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'processed'
)

def full_preprocessing_pipeline(nombre_dataset):
    os.makedirs(DATA_PROC, exist_ok=True)

    archivo_train = os.path.join(DATA_PROC, f'{nombre_dataset}_train.pkl')
    archivo_test  = os.path.join(DATA_PROC, f'{nombre_dataset}_test.pkl')

    if os.path.exists(archivo_train) and os.path.exists(archivo_test):
        print(f"[{nombre_dataset}] cargando datos procesados desde disco...")
        X_train, y_train = pd.read_pickle(archivo_train)
        X_test,  y_test  = pd.read_pickle(archivo_test)
        return X_train, X_test, y_train, y_test

    print(f"\n{'='*50}")
    print(f"Procesando: {nombre_dataset.upper()}")
    print(f"{'='*50}")

    if nombre_dataset == 'nslkdd':
        df_train, df_test = load_nslkdd()

    elif nombre_dataset == 'cicids2017':
        df_train, df_test = load_cicids2017()

    elif nombre_dataset == 'unswnb15':
        df_train, df_test = load_unswnb15()

    if nombre_dataset == 'cicids2017':
        df_train = clean_dataset(df_train, nombre_dataset)
    else:
        df_train = clean_dataset(df_train, f'{nombre_dataset}_train')
        df_test  = clean_dataset(df_test,  f'{nombre_dataset}_test')

    df_train = binary_label(df_train, nombre_dataset)
    if df_test is not None:
        df_test = binary_label(df_test, nombre_dataset)

    df_train = encode_categorical(df_train, nombre_dataset)
    if df_test is not None:
        df_test = encode_categorical(df_test, nombre_dataset)
        df_test = align_columns(df_train, df_test)

    X_train = df_train.drop(columns=['label'])
    y_train = df_train['label']

    if df_test is not None:
        X_test = df_test.drop(columns=['label'])
        y_test = df_test['label']

    if nombre_dataset == 'cicids2017':
        X_train, X_test, y_train, y_test = stratified_split(X_train, y_train, nombre_dataset)

        from training.hyperparams import CICIDS_SUBSAMPLE_SIZE
        from sklearn.model_selection import train_test_split as _tts

        if len(X_train) > CICIDS_SUBSAMPLE_SIZE:
            X_train, _, y_train, _ = _tts(
                X_train, y_train,
                train_size=CICIDS_SUBSAMPLE_SIZE,
                stratify=y_train,
                random_state=42
            )
            print(f"  [{nombre_dataset}] subsampleo estratificado: {len(X_train)} filas")

    features_seleccionadas = select_features_gini(X_train, y_train, nombre_dataset)
    X_train = X_train[features_seleccionadas]
    X_test  = X_test[features_seleccionadas]

    X_train, X_test = normalize_minmax(X_train, X_test, nombre_dataset)

    pd.to_pickle((X_train, y_train), archivo_train)
    pd.to_pickle((X_test,  y_test),  archivo_test)
    print(f"  [{nombre_dataset}] guardado en: {archivo_train}")

    return X_train, X_test, y_train, y_test
