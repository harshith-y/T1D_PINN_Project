"""
Comprehensive T1D PINN Training DAG
Trains all models on all patients
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.python import PythonOperator

default_args = {
    'owner': 'harsh',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    't1d_comprehensive_training',
    default_args=default_args,
    description='Train all models on all patients',
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['t1d', 'training', 'comprehensive'],
)

# Configuration
MODELS = ['birnn', 'pinn', 'modified_mlp']
PATIENTS = list(range(2, 12))  # Pat2-11

# Create tasks for each combination
tasks = []

for model in MODELS:
    for patient in PATIENTS:
        task_id = f'train_{model}_pat{patient}'
        
        task = DockerOperator(
            task_id=task_id,
            image='t1d-pinn:latest',
            command=f'python scripts/train_inverse.py --config configs/{model}_inverse.yaml --patient {patient}',
            docker_url='unix://var/run/docker.sock',
            network_mode='t1d-network',
            auto_remove=True,
            mounts=[
                {
                    'source': '/home/ubuntu/T1D_PINN_Project/data',
                    'target': '/data',
                    'type': 'bind',
                    'read_only': True,
                },
                {
                    'source': '/home/ubuntu/T1D_PINN_Project/results',
                    'target': '/results',
                    'type': 'bind',
                },
            ],
            environment={
                'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
            },
            dag=dag,
        )
        
        tasks.append(task)

# Optional: Set dependencies (e.g., run models sequentially per patient)
# for i in range(1, len(tasks)):
#     tasks[i].set_upstream(tasks[i-1])
