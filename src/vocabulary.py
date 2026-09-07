from json import JSONDecodeError, load
from pathlib import Path
from typing import Any


class Vocabulary:
    """Provide access to the model's token vocabulary."""

    def __init__(self, model: Any) -> None:
        """Initialize the vocabulary and load the model's token mappings."""

        self.model = model
        self.load_vocabulary()

    def load_vocabulary(self) -> None:
        """Load the token-to-ID mappings from the vocabulary JSON file."""

        self.id_to_token: dict[int, str] = {}
        self.token_to_id: dict[str, int] = {}

        vocab_path = self.model.get_path_to_vocab_file()
        path = Path(vocab_path)

        if not path.exists():
            raise FileNotFoundError(
                f"{vocab_path} doesn't exist"
            )

        if not path.is_file():
            raise FileNotFoundError(
                f"{vocab_path} is not a file"
            )

        if path.suffix != ".json":
            raise ValueError(
                f"{vocab_path} should be a JSON file"
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                data = load(file)
        except JSONDecodeError:
            raise ValueError("invalid JSON file")

        if not isinstance(data, dict):
            raise ValueError(
                "Vocabulary JSON must contain an object"
            )

        for token, token_id in data.items():
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    def get_id_from_token(self, token: str) -> int:
        """Return the token ID associated with a token."""

        return self.token_to_id[token]

    def get_token_from_id(self, token_id: int) -> str:
        """Return the token associated with a token ID."""

        return self.id_to_token[token_id]
