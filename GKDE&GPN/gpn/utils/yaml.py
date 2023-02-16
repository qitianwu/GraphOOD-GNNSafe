import os
import yaml


def validate_yaml_file(path: str, directory: str, file_name: str) -> str:
    """utility method which reads contents of a .yaml file

    Args:
        path (str): base path
        directory (str): directory containing .yaml files
        file_name (str): filename (without file extension)

    Raises:
        AssertionError: raised if path does not exist

    Returns:
        str: str-encoded file
    """

    file_name = file_name.lower()
    file_path = os.path.join(path, directory, f'{file_name}.yaml')
    if not os.path.exists(file_path):
        raise AssertionError(f'"{file_name}"" not found in {path}/{directory}!')

    return file_path


def read_yaml_file(path: str, directory: str, file_name: str) -> dict:
    """utility function which reads a .yaml file and returns its content as a dictionary

    Args:
        path (str): base path
        directory (str): directory containing .yaml files
        file_name (str): filename (without file extension)

    Returns:
        dict: contents of .yaml file
    """

    f = validate_yaml_file(path, directory, file_name)
    with open(f) as file:
        yaml_file = yaml.safe_load(file)

    if yaml_file is None:
        yaml_file = {}

    return yaml_file


def dump_yaml_file(dictionary: dict, path: str, directory: str, file_name: str) -> None:
    """write a dictionary to a .yaml file at the specified location

    Args:
        dictionary (dict): values to write
        path (str): base path
        directory (str): directory containing .yaml files
        file_name (str): filename (with file extension)
    """

    with open(os.path.join(path, directory, file_name), 'w') as file:
        yaml.dump(dictionary, file, default_flow_style=False, indent=2, sort_keys=False)
