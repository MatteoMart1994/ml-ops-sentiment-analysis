"""
Periodic MLOps loop: collect data → evaluate → optionally retrain → deploy.

Tasks exchange file paths and metrics through XCom; artifacts are written under
MLOPS_ARTIFACT_DIR. Collection, evaluation and training use fixed demo values.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

ARTIFACT_ROOT = Path(os.environ.get("MLOPS_ARTIFACT_DIR", "/tmp/airflow_sentiment"))
DATA_PATH = ARTIFACT_ROOT / "collected.json"
METRICS_PATH = ARTIFACT_ROOT / "metrics.json"
MODEL_PATH = ARTIFACT_ROOT / "model_artifact.json"

default_args = {
    "owner": "ml-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def collect_data(**context):
    """Write a batch of incoming texts to the artifact directory."""
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    batch = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "rows": [
            {"text": "Still waiting for my refund, very angry.", "source": "support_queue"},
            {"text": "Thanks for the quick fix!", "source": "support_queue"},
            {"text": "Invoice #4421 was issued yesterday.", "source": "tickets"},
        ],
    }
    DATA_PATH.write_text(json.dumps(batch, indent=2))
    context["ti"].xcom_push(key="dataset_path", value=str(DATA_PATH))
    context["ti"].xcom_push(key="row_count", value=len(batch["rows"]))
    return str(DATA_PATH)


def evaluate_model(**context):
    ti = context["ti"]
    dataset_path = Path(ti.xcom_pull(task_ids="collect_data", key="dataset_path"))
    payload = json.loads(dataset_path.read_text())
    metrics = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(payload["rows"]),
        "macro_f1": 0.71,
        "negative_rate_pred": 0.38,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    ti.xcom_push(key="metrics", value=metrics)
    return metrics


def choose_retrain_branch(**context):
    metrics = context["ti"].xcom_pull(task_ids="evaluate_model", key="metrics")
    if metrics["macro_f1"] < 0.75:
        return "train_model"
    return "skip_retrain"


def train_model(**context):
    metrics = context["ti"].xcom_pull(task_ids="evaluate_model", key="metrics")
    artifact = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "base_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "trigger_reason": f"macro_f1={metrics['macro_f1']} below threshold",
        "output_dir": str(MODEL_PATH),
    }
    MODEL_PATH.write_text(json.dumps(artifact, indent=2))
    context["ti"].xcom_push(key="model_artifact", value=artifact)
    return artifact


def deploy_model(**context):
    artifact = context["ti"].xcom_pull(task_ids="train_model", key="model_artifact")
    deployment = {
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "image_tag": "sentiment-inference:latest",
        "model_artifact": artifact,
        "status": "rolled_out_to_staging",
    }
    context["ti"].xcom_push(key="deployment", value=deployment)
    return deployment


with DAG(
    dag_id="sentiment_retrain_cycle",
    default_args=default_args,
    description="Collect → evaluate → (optional) retrain → deploy",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["sentiment", "mlops"],
) as dag:
    start = EmptyOperator(task_id="start")

    collect_data_task = PythonOperator(
        task_id="collect_data",
        python_callable=collect_data,
    )

    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    branch_retrain = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=choose_retrain_branch,
    )

    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    skip_retrain = EmptyOperator(task_id="skip_retrain")

    deploy_model_task = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model,
    )

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")

    start >> collect_data_task >> evaluate_model_task >> branch_retrain
    branch_retrain >> train_model_task >> deploy_model_task >> end
    branch_retrain >> skip_retrain >> end
