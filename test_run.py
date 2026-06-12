import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.pipeline import full_preprocessing_pipeline
from training.train import train_single_run
from evaluation.evaluate import load_model, predict_on_test_set, compute_metrics

TEST_TS = {'nslkdd': 100_000, 'cicids2017': 100_000, 'unswnb15': 100_000}

for ds in ['nslkdd', 'cicids2017', 'unswnb15']:

    X_train, X_test, y_train, y_test = full_preprocessing_pipeline(ds)

    model_path, _ = train_single_run('PPO', ds, 42, X_train, y_train, timesteps=TEST_TS[ds])
    model  = load_model('PPO', model_path)
    y_pred = predict_on_test_set(model, X_test.values)
    m = compute_metrics(y_test.values, y_pred)
    print(f"[{ds}] acc={m['accuracy']} f1={m['f1']} recall={m['recall']} fpr={m['fpr']}")

print("TEST DONE")
