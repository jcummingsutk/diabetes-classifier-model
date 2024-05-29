import os
from typing import Any

import yaml


def load_env_vars(config_filename: str, config_secret_filename: str) -> dict[str, Any]:
    with open(config_filename, "r") as f:
        config = yaml.safe_load(f)
    for key in config["azure"].keys():
        print(key)
        os.environ[key] = config["azure"][key]

    if os.path.exists(config_secret_filename):
        with open(config_secret_filename, "r") as fp:
            config_secret = yaml.safe_load(fp)
        for key in config_secret["azure"].keys():
            os.environ[key] = config_secret["azure"][key]
