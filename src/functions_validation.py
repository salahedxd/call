from json import JSONDecodeError, load
from pathlib import Path


VALID_KEYS = {"name", "description", "parameters", "returns"}
VALID_TYPES = {"number", "integer", "string", "boolean", "null"}


def functions_validator(file_path: str):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{file_path} doesn't exist")

    if not path.is_file():
        raise FileNotFoundError(f"{file_path} is not a file")

    if path.suffix != ".json":
        raise ValueError(f"{file_path} should be a JSON file")

    try:
        with path.open("r", encoding="utf-8") as file:
            functions = load(file)
    except JSONDecodeError:
        raise ValueError("invalid JSON file")

    if not isinstance(functions, list):
        raise TypeError("the root type must be a list")

    if not functions:
        raise ValueError("the functions can't be empty")

    for function in functions:
        if not isinstance(function, dict):
            raise TypeError("the function must be a dictionary")

        if set(function) != VALID_KEYS:
            raise ValueError("invalid function format")

        if not isinstance(function["name"], str):
            raise ValueError("the name must be a string")

        if not isinstance(function["description"], str):
            raise ValueError("the description must be a string")

        parameters = function["parameters"]

        if not isinstance(parameters, dict):
            raise TypeError("the parameters must be a dictionary")

        for name, parameter in parameters.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "the parameter name must be a non-empty string"
                )

            if not isinstance(parameter, dict):
                raise TypeError("the parameter must be a dictionary")

            if "type" not in parameter:
                raise ValueError("parameter missing type")

            if parameter["type"] not in VALID_TYPES:
                raise ValueError("invalid parameter type")

        returns = function["returns"]

        if not isinstance(returns, dict):
            raise TypeError("the return must be a dictionary")

        if "type" not in returns:
            raise ValueError("return missing type")

        if returns["type"] not in VALID_TYPES:
            raise ValueError("invalid return type")

    functions.append({
        "name": "fn_not_found",
        "description": (
            "Choose this function when the user's prompt "
            "does not match any available function."
        ),
        "parameters": {
            "a": {
                "type": "null"
            }
        },
        "returns": {
            "type": "null"
        }
    })

    return functions
