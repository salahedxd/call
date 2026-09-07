import json
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]


class SchemaState:
    START = 0

    EXPECT_PROMPT_KEY = 1
    EXPECT_PROMPT_COLON = 2
    EXPECT_PROMPT_VALUE = 3
    EXPECT_PROMPT_COMMA = 4

    EXPECT_NAME_KEY = 5
    EXPECT_NAME_COLON = 6
    EXPECT_FUNCTION_NAME = 7
    EXPECT_NAME_COMMA = 8

    EXPECT_PARAMETERS_KEY = 9
    EXPECT_PARAMETERS_COLON = 10
    EXPECT_PARAMETERS_OPEN = 11

    EXPECT_PARAMETER_NAME = 12
    EXPECT_PARAMETER_COLON = 13
    EXPECT_PARAMETER_VALUE = 14
    EXPECT_PARAMETER_OR_END = 15

    EXPECT_OBJECT_END = 16
    DONE = 17


class Schema:
    """Manage the structure and token constraints of a function call."""

    def __init__(
        self,
        functions: dict,
        model: Small_LLM_Model,
        prompt: str,
    ) -> None:
        """Initialize the schema with the available functions, model
        and prompt. and other attributes to track the state
        of the schema and the generated output."""

        self.functions: dict = functions
        self.model: Small_LLM_Model = model
        self.prompt: str = prompt

        self.quoted_spans: set[str] = self._extract_quoted_spans(prompt)

        self.state: int = SchemaState.START
        self.selected_function: str | None = None
        self.current_parameter: str | None = None

        self.expected_ids: list[int] = []
        self.expected_index: int = 0

        self.function_sequences: list[tuple[str, list[int]]] = []
        self.active_sequences: list[tuple[str, list[int]]] = []
        self.sequence_index: int = 0

        self.parameter_sequences: list[tuple[str, list[int]]] = []
        self.active_parameter_sequences: list[tuple[str, list[int]]] = []
        self.parameter_sequence_index: int = 0
        self.parameter_index: int = 0
        self.used_parameters: set[str] = set()

        self.value_type: str | None = None
        self.value_buffer: str = ""
        self.value_started: bool = False
        self.value_token_count: int = 0
        self.finished: bool = False

        self.max_string_tokens: int = 20
        self.safe_string_tokens: list[int] | None = None

    def select_token(
        self,
        logits: list[float],
        allowed_ids: list[int],
    ) -> int:
        """Select the best token from the allowed tokens
        based on the current schema state and logits."""

        is_string_value = (
            self.state == SchemaState.EXPECT_PARAMETER_VALUE
            and self.value_type == "string"
            and self.value_started
        )

        if not is_string_value:

            best_token_id = allowed_ids[0]
            best_score = logits[best_token_id]

            for token_id in allowed_ids:
                score = logits[token_id]

                if score > best_score:
                    best_token_id = token_id
                    best_score = score

            return best_token_id

        quote_id = self.encode('"')[0]
        content_ids = []

        for token_id in allowed_ids:
            if token_id != quote_id:
                content_ids.append(token_id)

        appeared_in_prompt = self._value_appeared_in_prompt()
        has_open_construct = (
            "[" in self.value_buffer
            or "(" in self.value_buffer
        )

        if (
            not appeared_in_prompt
            and not has_open_construct
            and self.value_token_count > 0
        ):
            return quote_id

        if (
            self.value_buffer
            and self.value_buffer in self.quoted_spans
            and not has_open_construct
        ):
            return quote_id

        safe_content_ids = []
        blocked_prompt_tokens = []

        for token_id in content_ids:
            text = self.model.decode([token_id])

            if text.lstrip().startswith("\\"):
                continue

            prospective = self.value_buffer + text

            if (
                self.value_buffer.isalnum()
                and text[0] in ".*+?\\"
            ):
                continue

            if (
                self.value_token_count == 0
                and text.lstrip().startswith("(")
                and prospective not in self.prompt
            ):
                continue

            if (
                self.value_buffer
                and appeared_in_prompt
                and prospective not in self.prompt
                and not text.lstrip().startswith("[")
            ):
                blocked_prompt_tokens.append(token_id)
                continue

            safe_content_ids.append(token_id)

        if not safe_content_ids:

            if (
                blocked_prompt_tokens
                and self.value_token_count > 0
            ):
                return quote_id

            safe_content_ids = content_ids

        best_content = max(safe_content_ids, key=lambda t: logits[t])

        if quote_id in allowed_ids and self.value_token_count > 0:
            balanced = self._has_balanced_brackets()

            if balanced:
                if self._should_extend_regex(best_content):
                    return best_content
                return quote_id

            quote_score = logits[quote_id]
            best_other_score = float("-inf")

            for token_id in allowed_ids:

                score = logits[token_id]

                if score > best_other_score:
                    best_other_score = score

            if quote_score >= best_other_score:
                return quote_id

        return best_content

    def _has_balanced_brackets(self) -> bool:
        """check if the value buffer has balanced brackets and parentheses."""

        buf = self.value_buffer

        if "[" not in buf and "(" not in buf:
            return False

        count = 0

        for char in buf:
            if char in "([":
                count += 1
            elif char in ")]":
                count -= 1

        if count != 0:
            return False

        return True

    def _value_appeared_in_prompt(self) -> bool:
        """check if the value buffer appears in the prompt."""

        return self.value_buffer in self.prompt

    @staticmethod
    def _extract_quoted_spans(prompt: str) -> set[str]:
        """Extract quoted spans from the prompt
        to help with string value validation."""

        spans = set()

        start = prompt.find('"')
        end = prompt.find('"', start + 1)

        if start != -1 and end != -1:
            spans.add(prompt[start + 1:end])

        start = prompt.find("'")
        end = prompt.find("'", start + 1)

        if start != -1 and end != -1:
            spans.add(prompt[start + 1:end])

        return spans

    def _should_extend_regex(self, candidate_id: int) -> bool:
        """Check if the candidate token should extend a regex pattern."""

        text = self.model.decode([candidate_id])
        quantifiers = ["*", "+", "?", "{"]

        return text in quantifiers

    def encode(self, text: str) -> list[int]:
        """Encode text into token IDs using the model's tokenizer."""
        return list(self.model.encode(text).squeeze(0).tolist())

    def allowed_tokens(self) -> list[int]:
        """Determine the allowed tokens based on the current schema state."""

        if self.state == SchemaState.EXPECT_FUNCTION_NAME:
            if not self.active_sequences:
                self.load_function_sequences()
            return self.allowed_function_tokens()

        if self.state == SchemaState.EXPECT_PARAMETER_NAME:
            if not self.active_parameter_sequences:
                self.load_parameter_sequences()
            return self.allowed_parameter_tokens()

        if self.state == SchemaState.EXPECT_PARAMETER_VALUE:
            if self.value_type is None:
                self.load_parameter_value()
            return self.allowed_value_tokens()

        if self.state == SchemaState.EXPECT_PARAMETER_OR_END:
            return self.allowed_parameter_end_tokens()

        if not self.expected_ids:
            self.load_expected()

        return [self.expected_ids[self.expected_index]]

    def load_expected(self) -> None:
        """Load the expected token IDs based on the current schema state."""

        expected = None

        if self.state == SchemaState.START:
            expected = "{"
        elif self.state == SchemaState.EXPECT_PROMPT_KEY:
            expected = '"prompt"'
        elif self.state == SchemaState.EXPECT_PROMPT_COLON:
            expected = ":"
        elif self.state == SchemaState.EXPECT_PROMPT_VALUE:
            expected = json.dumps(self.prompt)
        elif self.state == SchemaState.EXPECT_PROMPT_COMMA:
            expected = ","
        elif self.state == SchemaState.EXPECT_NAME_KEY:
            expected = '"name"'
        elif self.state == SchemaState.EXPECT_NAME_COLON:
            expected = ":"
        elif self.state == SchemaState.EXPECT_NAME_COMMA:
            expected = ","
        elif self.state == SchemaState.EXPECT_PARAMETERS_KEY:
            expected = '"parameters"'
        elif self.state == SchemaState.EXPECT_PARAMETERS_COLON:
            expected = ":"
        elif self.state == SchemaState.EXPECT_PARAMETERS_OPEN:
            expected = "{"
        elif self.state == SchemaState.EXPECT_PARAMETER_COLON:
            expected = ":"
        elif self.state == SchemaState.EXPECT_OBJECT_END:
            expected = "}"

        if expected is not None:
            self.expected_ids = self.encode(expected)
            self.expected_index = 0

    def load_function_sequences(self) -> None:
        """Load the token sequences for all available function names."""

        self.function_sequences = []

        for name in self.functions:
            tokens = self.encode(f'"{name}"')
            self.function_sequences.append((name, tokens))

        self.active_sequences = self.function_sequences.copy()
        self.sequence_index = 0

    def allowed_function_tokens(self) -> list[int]:
        """Determine the allowed tokens for function names
        based on the current schema state."""

        allowed = []

        for name, sequence in self.active_sequences:
            if self.sequence_index < len(sequence):
                token_id = sequence[self.sequence_index]

                allowed.append(token_id)

        return allowed

    def load_parameter_sequences(self) -> None:
        """Load the token sequences
        for all parameters of the selected function."""

        if self.selected_function is None:
            raise ValueError("No function selected")

        function = self.functions[self.selected_function]

        self.parameter_sequences = []

        for name in function.parameters:
            if name not in self.used_parameters:
                tokens = self.encode(f'"{name}"')
                self.parameter_sequences.append((name, tokens))

        self.active_parameter_sequences = self.parameter_sequences.copy()
        self.parameter_sequence_index = 0

    def allowed_parameter_tokens(self) -> list[int]:
        """determine the allowed tokens for parameter names
        based on the current schema state."""

        allowed = []

        for name, sequence in self.active_parameter_sequences:
            if self.parameter_sequence_index < len(sequence):

                token_id = sequence[self.parameter_sequence_index]

                allowed.append(token_id)

        return allowed

    def load_parameter_value(self) -> None:
        """Load the expected value type for the current parameter
        based on the selected function's definition."""

        function = self.functions[self.selected_function]
        parameter = function.parameters[self.current_parameter]

        self.value_type = parameter.type
        self.value_buffer = ""
        self.value_token_count = 0
        self.value_started = False

    def allowed_value_tokens(self) -> list[int]:
        """Determine the allowed tokens for parameter values"""

        if self.value_type == "boolean":
            return self.encode("true") + self.encode("false")

        if self.value_type == "null":
            return self.encode("null")

        if self.value_type in ("number", "integer"):
            prompt_tokens = self.get_prompt_number_tokens()

            if not prompt_tokens:
                raise ValueError("No number found in prompt")

            if self.value_token_count < len(prompt_tokens):
                return [prompt_tokens[self.value_token_count]]

            return self.allowed_parameter_end_tokens()

        if self.value_type == "string":
            quote_id = self.encode('"')[0]

            if not self.value_started:
                return [quote_id]

            if self.value_token_count == self.max_string_tokens:
                return [quote_id]

            content_tokens = self.get_string_content_tokens()

            if self.value_token_count == 0:
                return content_tokens

            return content_tokens + [quote_id]

        raise ValueError(
            f"Unsupported parameter type: {self.value_type}"
        )

    def get_prompt_number_tokens(self) -> list[int]:
        """Find the numbers in the prompt
        and encode the number for the current parameter."""

        numbers = []

        for word in self.prompt.split():
            word = word.strip(".,!?")

            try:
                float(word)
                numbers.append(word)
            except ValueError:
                continue

        function = self.functions[self.selected_function]
        parameter_count = len(function.parameters)

        if len(numbers) != parameter_count:
            raise ValueError(
                "Number of numbers in prompt does not match parameter count"
            )

        return self.encode(numbers[self.parameter_index])

    def allowed_parameter_end_tokens(self) -> list[int]:
        """Return the tokens allowed after a parameter value."""

        comma_id = self.encode(",")[0]
        close_id = self.encode("}")[0]

        parameter_count = len(
            self.functions[self.selected_function].parameters
        )

        used_count = len(self.used_parameters)

        allowed = []

        if used_count < parameter_count:
            allowed.append(comma_id)

        if used_count == parameter_count:
            allowed.append(close_id)

        return allowed

    def get_string_content_tokens(self) -> list[int]:
        """Build and cache the token IDs
        that are safe for JSON string content."""

        if self.safe_string_tokens is not None:
            return self.safe_string_tokens

        quote_id = self.encode('"')[0]
        allowed = []

        for token_id in range(self.model._tokenizer.vocab_size):
            if token_id == quote_id:
                continue

            token_text = self.model.decode([token_id])

            if '"' in token_text:
                if '\\"' not in token_text:
                    continue

            if "\\" in token_text and token_text not in (
                "\\\\", "\\\"", "\\/", "\\b", "\\f", "\\n", "\\r", "\\t",
            ):
                continue

            allowed.append(token_id)

        self.safe_string_tokens = allowed
        return self.safe_string_tokens
