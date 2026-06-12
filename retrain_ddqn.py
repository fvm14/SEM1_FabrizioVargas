"""
Reentrena los 9 runs de DDQN (3 datasets x 3 semillas) con la implementacion
Double DQN real (training/double_dqn.py) y actualiza metrics_s1.csv.

Los runs de PPO y DQN no se tocan: se conservan sus filas en el CSV y sus
modelos/curvas en disco. Antes de sobreescribir, se respalda el CSV de
metricas y las curvas DDQN anteriores con sufijo _vanilla_backup.
"""
import sys
import os
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from preprocessing.pipeline import full_preprocessing_pipeline
from training.train import train_single_run
from training.hyperparams import SEEDS, DATASETS, TIMESTEPS_PER_DATASET
from evaluation.evaluate import load_model, predict_on_test_set, compute_metrics, print_summary

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(BASE_DIR, 'results', 'metrics', 'metrics_s1.csv')
CURVES_DIR = os.path.join(BASE_DIR, 'results', 'convergence_curves')

def backup_previous_ddqn():

    if os.path.exists(METRICS_PATH):
        shutil.copy2(METRICS_PATH, METRICS_PATH.replace('.csv', '_vanilla_backup.csv'))
        print(f"[backup] {METRICS_PATH} -> metrics_s1_vanilla_backup.csv")

    for fname in os.listdir(CURVES_DIR):
        if fname.startswith('DDQN_') and fname.endswith('_curve.csv') and 'backup' not in fname:
            src = os.path.join(CURVES_DIR, fname)
            dst = os.path.join(CURVES_DIR, fname.replace('_curve.csv', '_curve_vanilla_backup.csv'))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    print("[backup] curvas DDQN anteriores respaldadas con sufijo _vanilla_backup")

def main():
    print("=" * 55)
    print("Reentrenamiento DDQN (Double DQN real) -- 9 runs")
    print("=" * 55)

    backup_previous_ddqn()

    datasets = {}
    for ds in DATASETS:
        X_train, X_test, y_train, y_test = full_preprocessing_pipeline(ds)
        datasets[ds] = (X_train, X_test, y_train, y_test)

    partial_path = os.path.join(BASE_DIR, 'results', 'metrics', 'metrics_ddqn_partial.csv')
    new_rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            X_train, X_test, y_train, y_test = datasets[dataset]

            run_id = f"DDQN_{dataset.upper()}_seed{seed}"
            model_path = os.path.join(BASE_DIR, 'models', run_id)

            if os.path.exists(model_path + '.zip'):
                print(f"\n[skip] {run_id}: modelo ya existe, solo evaluo")
            else:
                model_path, _ = train_single_run(
                    agent_name='DDQN',
                    dataset_name=dataset,
                    seed=seed,
                    X_train=X_train,
                    y_train=y_train,
                    timesteps=TIMESTEPS_PER_DATASET[dataset]
                )

            model = load_model('DDQN', model_path)
            y_pred = predict_on_test_set(model, X_test.values)
            metrics = compute_metrics(y_test.values, y_pred)

            row = {'agent': 'DDQN', 'dataset': dataset, 'seed': seed, **metrics}
            new_rows.append(row)

            pd.DataFrame(new_rows).to_csv(partial_path, index=False)
            print(f"  metrics: acc={metrics['accuracy']} f1={metrics['f1']} "
                  f"recall={metrics['recall']} fpr={metrics['fpr']}")

    df_old = pd.read_csv(METRICS_PATH)
    df_keep = df_old[df_old['agent'] != 'DDQN']
    df_new = pd.concat([df_keep, pd.DataFrame(new_rows)], ignore_index=True)
    df_new.to_csv(METRICS_PATH, index=False)
    print(f"\n[OK] metrics_s1.csv actualizado ({len(new_rows)} filas DDQN reemplazadas)")

    print_summary(df_new)
    print("\n[OK] Reentrenamiento DDQN completo.")

if __name__ == '__main__':
    main()
