import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from stable_baselines3 import PPO, DQN
from training.double_dqn import DoubleDQN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(agent_name, model_path):
    if agent_name == 'PPO':
        return PPO.load(model_path)
    elif agent_name == 'DDQN':
        return DoubleDQN.load(model_path)
    return DQN.load(model_path)

def predict_on_test_set(model, X_test):

    observations = X_test.astype(np.float32)
    actions, _ = model.predict(observations, deterministic=True)
    return actions.astype(int)

def compute_metrics(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    if (fp + tn) > 0:
        false_positive_rate = fp / (fp + tn)
    else:
        false_positive_rate = 0.0

    results = {
        'accuracy':  round(accuracy_score(y_true, y_pred), 6),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 6),
        'recall':    round(recall_score(y_true, y_pred, zero_division=0), 6),
        'f1':        round(f1_score(y_true, y_pred, zero_division=0), 6),
        'fpr':       round(false_positive_rate, 6),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

    return results

def evaluate_all_models(datasets_dict, run_log):
    all_results = []

    for entry in run_log:
        agent_name  = entry['agent']
        dataset_name = entry['dataset']
        seed        = entry['seed']

        print(f"Evaluando: {agent_name}_{dataset_name.upper()}_seed{seed}")

        X_train, X_test, y_train, y_test = datasets_dict[dataset_name]

        model  = load_model(agent_name, entry['model_path'])
        y_pred = predict_on_test_set(model, X_test.values)
        metrics = compute_metrics(y_test.values, y_pred)

        all_results.append({
            'agent':   agent_name,
            'dataset': dataset_name,
            'seed':    seed,
            **metrics
        })

    results_df = pd.DataFrame(all_results)

    output_path = os.path.join(BASE_DIR, 'results', 'metrics', 'metrics_s1.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n[OK] Metricas guardadas en: {output_path}")
    return results_df

def print_summary(df_metrics):

    summary = df_metrics.groupby(['agent', 'dataset'])[
        ['accuracy', 'precision', 'recall', 'f1', 'fpr']
    ].agg(['mean', 'std']).round(4)

    print("\n" + "=" * 80)
    print("RESUMEN DE RESULTADOS - Seminario 1")
    print("=" * 80)
    print(summary.to_string())
