from json import JSONDecodeError, load
from pathlib import Path
from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, ValidationError


class ParameterDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["number", "integer", "string", "boolean", "null"]


class ReturnDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["number", "integer", "string", "boolean", "null"]


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: Dict[str, ParameterDefinition]
    returns: ReturnDefinition


def functions_validator(file_path: str) -> List[dict]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{file_path} doesn't exist")

    if not path.is_file():
        raise FileNotFoundError(f"{file_path} is not a file")

    if path.suffix != ".json":
        raise ValueError(f"{file_path} should be a JSON file")

    try:
        with path.open("r", encoding="utf-8") as file:
            raw_functions = load(file)
    except JSONDecodeError:
        raise ValueError("invalid JSON file")

    if not isinstance(raw_functions, list):
        raise TypeError("the root type must be a list")

    if not raw_functions:
        raise ValueError("the functions can't be empty")

    functions = []

    for item in raw_functions:
        try:
            function = FunctionDefinition.model_validate(item)
            functions.append(function)
        except ValidationError as e:
            raise ValueError(f"invalid function format: {e}")

    result = [function.model_dump() for function in functions]

    result.append({
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

    return result
