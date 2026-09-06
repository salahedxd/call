from json import JSONDecodeError, load
from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict, ValidationError


class PromptItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str


def prompt_validator(file_path: str) -> List[dict]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"{file_path} doesn't exist")

    if not path.is_file():
        raise FileNotFoundError(f"{file_path} is not a file")

    if path.suffix != ".json":
        raise ValueError(f"{file_path} should be a JSON file")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = load(file)
    except JSONDecodeError:
        raise ValueError("invalid JSON file")

    if not isinstance(data, list):
        raise ValueError("JSON root must be a list")

    if not data:
        raise ValueError("JSON file cannot be empty")

    items = []

    for item in data:
        try:
            prompt = PromptItem.model_validate(item)
            items.append(prompt)
        except ValidationError as e:
            raise ValueError(f"invalid prompt format: {e}")

    return [item.model_dump() for item in items]
