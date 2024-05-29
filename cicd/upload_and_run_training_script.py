import os

from azure.ai.ml import Input, MLClient, command
from azure.identity import EnvironmentCredential

from diabetes_model_code.config import load_env_vars

load_env_vars("config.yaml", "config_secret.yaml")
cred = EnvironmentCredential()
ml_client = MLClient(
    subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_ML_RESOURCE_GROUP_NAME"],
    workspace_name=os.environ["AZURE_ML_WORKSPACE_NAME"],
    credential=cred,
)

command_job = command(
    code=os.path.join("diabetes_model_code"),
    command="python -m train_model_script --data-source ${{inputs.data}} --num-hyperopt-evals 100 --num-hyperopt-trials-to-log 5",
    environment=f"{os.environ['AZURE_ML_TRAINING_ENV_NAME']}:2",
    inputs={
        "data": Input(
            type="uri_folder",
            path="azureml://subscriptions/94f3bfe4-d65b-4af2-959a-f4cc3f4fef6a/resourcegroups/diabetes-classifier-2024/workspaces/diabetes-classifier-2024-dev/datastores/2024devdata/paths/data/",
        ),
    },
    compute=os.environ["AZURE_ML_TRAINING_COMPUTE_NAME"],
    experiment_name="diabetes_prediction",
)
returned_job = ml_client.jobs.create_or_update(command_job)
ml_client.jobs.stream(returned_job.name)
