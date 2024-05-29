import os

import yaml
from azure.ai.ml import Input, MLClient, command
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import EnvironmentCredential
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.environment import Environment

from diabetes_model.code.config import load_env_vars


def is_environ_updated(ml_client: MLClient) -> bool:
    try:
        env = ml_client.environments._get_latest_version(
            os.environ["AZURE_ML_TRAINING_ENV_NAME"]
        )
    except ResourceNotFoundError:
        return False
    with open(os.environ["CONDA_TRAINING_ENV_FILE"], "r") as f:
        local_conda_dict = yaml.safe_load(f)
    if local_conda_dict != env.conda_file:
        return False
    return True


def update_environ():
    sp_auth = ServicePrincipalAuthentication(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        service_principal_id=os.environ["AZURE_CLIENT_ID"],
        service_principal_password=os.environ["AZURE_CLIENT_SECRET"],
    )
    ws = Workspace(
        subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"],
        resource_group=os.environ["AZURE_ML_RESOURCE_GROUP_NAME"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE_NAME"],
        auth=sp_auth,
    )
    env = Environment.from_conda_specification(
        name=os.environ["AZURE_ML_TRAINING_ENV_NAME"],
        file_path=os.environ["CONDA_TRAINING_ENV_FILE"],
    )
    build_details = env.build(workspace=ws)
    build_details.wait_for_completion()


def main():
    load_env_vars("config.yaml", "config_secret.yaml")
    cred = EnvironmentCredential()
    ml_client = MLClient(
        subscription_id=os.environ["AZURE_ML_SUBSCRIPTION_ID"],
        resource_group_name=os.environ["AZURE_ML_RESOURCE_GROUP_NAME"],
        workspace_name=os.environ["AZURE_ML_WORKSPACE_NAME"],
        credential=cred,
    )
    print("checking environment")
    is_env_updated = is_environ_updated(ml_client)
    if not is_env_updated:
        print("environment needs updating, updating environment")
        update_environ()
    else:
        print("environment already up to date")

    env = ml_client.environments._get_latest_version(
        os.environ["AZURE_ML_TRAINING_ENV_NAME"]
    )
    env_version = env.version

    num_hyperopt_evals = os.environ["NUM_HYPEROPT_EVALS"]
    num_hyperopt_trials_to_log = os.environ["NUM_HYPEROPT_TRIALS_TO_LOG"]

    print("submitting training job")
    command_job = command(
        code=os.path.join("diabetes_model"),
        command=f"python -m code.train_model_script --data-source ${{inputs.data}} --num-hyperopt-evals {num_hyperopt_evals} --num-hyperopt-trials-to-log {num_hyperopt_trials_to_log}",
        environment=f"{os.environ['AZURE_ML_TRAINING_ENV_NAME']}:{env_version}",
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


if __name__ == "__main__":
    main()
