import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing.pipeline import full_preprocessing_pipeline
from envs.ids_env import IDSEnv
from stable_baselines3.common.env_checker import check_env
from training.train import run_all_experiments
from evaluation.evaluate import evaluate_all_models, print_summary
from training.hyperparams import DATASETS

def main():
    print("="*55)
    print("IDS-DRL System -- Seminario 1")
    print("="*55)

    datasets = {}
    for ds in DATASETS:
        X_train, X_test, y_train, y_test = full_preprocessing_pipeline(ds)
        datasets[ds] = (X_train, X_test, y_train, y_test)

    for ds in DATASETS:
        X_train, _, y_train, _ = datasets[ds]
        env = IDSEnv(X_train.values, y_train.values)
        check_env(env, warn=True, skip_render_check=True)
        env.close()
        print(f"[OK] env {ds}: check_env passed")

    run_log = run_all_experiments(datasets)

    df_metrics = evaluate_all_models(datasets, run_log)
    print_summary(df_metrics)

    print("\n[OK] Seminario 1 complete.")

if __name__ == '__main__':
    main()
