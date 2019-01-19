import yaml


def load_config(path: str) -> dict:
    """
    load yaml configuration file
    :param path: path for the yaml config file
    :return: config dictionary
    """

    with open(path, 'r') as file:
        try:
            cfg = yaml.safe_load(file)
        except yaml.YAMLError as err:
            print(err)

    return cfg
