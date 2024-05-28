import os

from azure.storage.blob import BlobServiceClient

from diabetes_model_code.config import load_env_vars

if __name__ == "__main__":
    load_env_vars("config.yaml", "config_secret.yaml")
    blob_service_client = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"]
    )
    container_client = blob_service_client.get_container_client(
        os.environ["BLOB_CONTAINER_NAME"]
    )
    blob_names = [name_ for name_ in container_client.list_blob_names()]
    for blob_name in blob_names:
        dir_name = os.path.dirname(blob_name)
        os.makedirs(dir_name, exist_ok=True)
        stream = container_client.download_blob(blob_name)
        with open(blob_name, "wb") as blob_file:
            blob_file.write(stream.readall())
