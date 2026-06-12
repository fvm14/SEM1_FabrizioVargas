import os
import glob
import pandas as pd

DATA_RAW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '..', 'data'
)

NSL_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

def load_nslkdd():
    train_path = os.path.join(DATA_RAW, 'NSL-KDD', 'KDDTrain+.txt')
    test_path  = os.path.join(DATA_RAW, 'NSL-KDD', 'KDDTest+.txt')

    df_train = pd.read_csv(train_path, header=None, names=NSL_COLUMNS)
    df_test  = pd.read_csv(test_path,  header=None, names=NSL_COLUMNS)

    df_train.drop(columns=['difficulty'], inplace=True)
    df_test.drop(columns=['difficulty'],  inplace=True)

    return df_train, df_test

def load_cicids2017():

    pattern = os.path.join(DATA_RAW, 'CICIDS2017', '*.csv')
    files   = glob.glob(pattern)

    dataframes = []
    for file in sorted(files):
        df = pd.read_csv(file, low_memory=False)

        df.columns = df.columns.str.strip()
        dataframes.append(df)

    df_all = pd.concat(dataframes, ignore_index=True)
    return df_all, None

def load_unswnb15():
    train_path = os.path.join(DATA_RAW, 'UNSW-NB15', 'UNSW_NB15_training-set.csv')
    test_path  = os.path.join(DATA_RAW, 'UNSW-NB15', 'UNSW_NB15_testing-set.csv')

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    for df in [df_train, df_test]:
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)
        if 'attack_cat' in df.columns:
            df.drop(columns=['attack_cat'], inplace=True)

    return df_train, df_test
