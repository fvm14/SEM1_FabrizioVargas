import numpy as np

def clean_dataset(df, nombre):
    """
    Limpia el dataset en tres pasos: eliminar duplicados, reemplazar infinitos
    por NaN, y eliminar filas con valores faltantes. Al final reporta cuanto
    se elimino y como queda la distribucion de clases.

    El reporte cuantitativo (% eliminado + distribucion) es buena practica
    documentada en la literatura de IDS para garantizar transparencia del
    preprocesamiento (Almuhanna y Dardouri, 2025; Tavallaee et al., 2009).
    """
    filas_iniciales = len(df)
    print(f"  [{nombre}] filas originales: {filas_iniciales}")

    df = df.drop_duplicates()
    print(f"  [{nombre}] tras eliminar duplicados: {len(df)}")

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.dropna()
    print(f"  [{nombre}] tras eliminar NaN: {len(df)}")

    porcentaje_eliminado = (filas_iniciales - len(df)) / filas_iniciales * 100
    print(f"  [{nombre}] eliminado total: {porcentaje_eliminado:.2f}%")

    columna_etiqueta = None
    if 'label' in df.columns:
        columna_etiqueta = 'label'
    elif 'Label' in df.columns:
        columna_etiqueta = 'Label'

    if columna_etiqueta is not None:
        conteo = df[columna_etiqueta].value_counts()
        print(f"  [{nombre}] distribucion de clases:")
        for clase, n in conteo.items():
            porcentaje = n / len(df) * 100

            clase_safe = str(clase).encode('ascii', 'replace').decode('ascii')
            print(f"    {clase_safe}: {n} ({porcentaje:.2f}%)")

    return df

def binary_label(df, nombre_dataset):
    """
    Convierte las etiquetas originales a formato binario:
    0 = trafico normal
    1 = ataque (cualquier tipo)
    """
    df = df.copy()

    if nombre_dataset == 'nslkdd':

        df['label'] = df['label'].apply(
            lambda x: 0 if str(x).strip() == 'normal' else 1
        )

    elif nombre_dataset == 'cicids2017':

        label_col = 'Label' if 'Label' in df.columns else ' Label'
        df['label'] = df[label_col].apply(
            lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1
        )
        df.drop(columns=[label_col], inplace=True)

    elif nombre_dataset == 'unswnb15':

        assert set(df['label'].unique()).issubset({0, 1}),            "UNSW-NB15: se encontraron valores distintos a 0 y 1 en la columna label"

    return df
