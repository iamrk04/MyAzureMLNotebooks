from typing import Dict, Any, List

import mlflow
from azure.ai.ml import MLClient


def _download_outputs(
    ml_client,
    job_name: str,
    download_path: str,
    output_name: str = None,
    all: bool = False,
) -> None:
    """
    Download the outputs from the azureml job.

    :param ml_client: Azureml workspace ml client.
    :param job_name: Name of the azureml job.
    :param download_path: Path to download the outputs.
    :param output_name: Name of the output to download.
    :param all: Flag to download all the outputs.
    :return: None.
    """
    ml_client.jobs.download(
        name=job_name,
        output_name=output_name,
        download_path=download_path,
        all=all,
    )


def _get_runs(ws_ml_client: MLClient, job_name: str, exp_name: str) -> List[mlflow.entities.Run]:
    """
    Get the runs from the mlflow for an azureml job.

    :param ws_ml_client: Azureml workspace ml client.
    :param job_name: Name of the azureml job.
    :param exp_name: Name of the azureml experiment.
    :return: List of runs.
    """

    # get and set the mlflow tracking uri
    mlflow_tracking_uri = ws_ml_client.workspaces.get(
        ws_ml_client.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # get the runs
    filter = f"tags.mlflow.parentRunId='{job_name}'"
    runs = mlflow.search_runs(
        experiment_names=[exp_name], filter_string=filter, output_format="list"
    )
    return runs


def get_mlflow_logged_metrics(ws_ml_client: MLClient, job_name: str, exp_name: str) -> Dict[str, Any]:
    """
    Get the logged metrics from the mlflow run for an azureml job.

    :param ws_ml_client: Azureml workspace ml client.
    :param job_name: Name of the azureml job.
    :param exp_name: Name of the azureml experiment.
    :return: Logged metrics.
    """
    runs = _get_runs(ws_ml_client, job_name, exp_name)
    logged_metrics = runs[-1].data.metrics
    return logged_metrics
