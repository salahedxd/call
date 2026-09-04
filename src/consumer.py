from .schema import SchemaState


class SchemaConsumer:
    """Handle token consumption and state transitions for Schema."""

    def __init__(self, schema):
        self.schema = schema

    def consume(self, token_id):

        if self.schema.state == SchemaState.EXPECT_FUNCTION_NAME:
            self.consume_function_token(token_id)
            return

        if self.schema.state == SchemaState.EXPECT_PARAMETER_NAME:
            self.consume_parameter_token(token_id)
            return

        if self.schema.state == SchemaState.EXPECT_PARAMETER_VALUE:
            self.consume_value_token(token_id)
            return

        if self.schema.state == SchemaState.EXPECT_PARAMETER_OR_END:
            self.consume_parameter_end(token_id)
            return

        self.schema.expected_index += 1

        if self.schema.expected_index == len(self.schema.expected_ids):
            self.schema.expected_ids = []
            self.schema.expected_index = 0
            self.advance_state()

    def consume_function_token(self, token_id):
        schema = self.schema
        valid_sequences = []

        for name, sequence in schema.active_sequences:
            if schema.sequence_index < len(sequence):
                if sequence[schema.sequence_index] == token_id:
                    valid_sequences.append((name, sequence))

        schema.active_sequences = valid_sequences

        if not schema.active_sequences:
            raise ValueError("Invalid function name token")

        schema.sequence_index += 1

        completed = []

        for name, sequence in schema.active_sequences:
            if schema.sequence_index == len(sequence):
                completed.append((name, sequence))

        if completed:
            schema.selected_function = completed[0][0]
            schema.function_sequences = []
            schema.active_sequences = []
            schema.sequence_index = 0
            self.advance_state()

    def consume_parameter_token(self, token_id):
        schema = self.schema
        valid_sequences = []

        for name, sequence in schema.active_parameter_sequences:
            if schema.parameter_sequence_index < len(sequence):
                if sequence[schema.parameter_sequence_index] == token_id:
                    valid_sequences.append((name, sequence))

        schema.active_parameter_sequences = valid_sequences

        if not schema.active_parameter_sequences:
            raise ValueError("Invalid parameter name token")

        schema.parameter_sequence_index += 1

        completed = []

        for name, sequence in schema.active_parameter_sequences:
            if schema.parameter_sequence_index == len(sequence):
                completed.append((name, sequence))

        if completed:
            schema.current_parameter = completed[0][0]
            schema.used_parameters.add(schema.current_parameter)

            schema.parameter_sequences = []
            schema.active_parameter_sequences = []
            schema.parameter_sequence_index = 0

            self.advance_state()

    def consume_parameter_end(self, token_id):
        schema = self.schema

        comma_id = schema.encode(",")[0]
        close_id = schema.encode("}")[0]

        if token_id == comma_id:
            schema.state = SchemaState.EXPECT_PARAMETER_NAME
            schema.expected_ids = []
            schema.expected_index = 0
            schema.parameter_sequences = []
            schema.active_parameter_sequences = []
            schema.parameter_sequence_index = 0
            return

        if token_id == close_id:
            schema.state = SchemaState.EXPECT_OBJECT_END
            schema.expected_ids = []
            schema.expected_index = 0
            return

        raise ValueError(
            f"Expected parameter separator or object end, got {token_id}"
        )

    def consume_value_token(self, token_id):
        schema = self.schema
        token_text = schema.model.decode([token_id])

        if schema.value_type == "string":
            quote_id = schema.encode('"')[0]
            escaped_quote_id = schema.encode('\\"')[0]

            if not schema.value_started:
                if token_id != quote_id:
                    raise ValueError(
                        "String value must start with a quote"
                    )

                schema.value_started = True
                return

            if token_id == escaped_quote_id:
                schema.value_buffer += '"'
                schema.value_token_count += 1
                return

            if token_id == quote_id:
                schema.value_type = None
                schema.value_started = False
                schema.value_buffer = ""
                schema.value_token_count = 0
                self.advance_state()
                return

            schema.value_buffer += token_text
            schema.value_token_count += 1
            return

        if schema.value_type == "boolean":
            valid_ids = schema.encode("true") + schema.encode("false")

            if token_id not in valid_ids:
                raise ValueError("Invalid boolean value")

            schema.value_type = None
            schema.value_buffer = ""
            self.advance_state()
            return

        if schema.value_type == "null":
            if token_id not in schema.encode("null"):
                raise ValueError("Invalid null value")

            schema.value_type = None
            schema.value_buffer = ""
            self.advance_state()
            return

        if schema.value_type in ("number", "integer"):
            comma_id = schema.encode(",")[0]
            close_id = schema.encode("}")[0]

            if token_id == comma_id:
                if not schema.value_buffer:
                    raise ValueError("Number cannot be empty")

                schema.value_type = None
                schema.state = SchemaState.EXPECT_PARAMETER_OR_END
                schema.expected_ids = []
                schema.expected_index = 0
                self.consume_parameter_end(token_id)
                return

            if token_id == close_id:
                if not schema.value_buffer:
                    raise ValueError("Number cannot be empty")

                schema.value_type = None
                schema.value_buffer = ""
                schema.value_token_count = 0
                schema.state = SchemaState.EXPECT_OBJECT_END
                schema.expected_ids = []
                schema.expected_index = 0
                return

            if not token_text:
                raise ValueError("Invalid number token")

            if not all(
                char in "0123456789.-+eE"
                for char in token_text
            ):
                raise ValueError("Invalid number token")

            schema.value_buffer += token_text
            schema.value_token_count += 1

    def advance_state(self):

        transitions = {
            SchemaState.START: SchemaState.EXPECT_PROMPT_KEY,
            SchemaState.EXPECT_PROMPT_KEY: SchemaState.EXPECT_PROMPT_COLON,
            SchemaState.EXPECT_PROMPT_COLON: SchemaState.EXPECT_PROMPT_VALUE,
            SchemaState.EXPECT_PROMPT_VALUE: SchemaState.EXPECT_PROMPT_COMMA,
            SchemaState.EXPECT_PROMPT_COMMA: SchemaState.EXPECT_NAME_KEY,
            SchemaState.EXPECT_NAME_KEY: SchemaState.EXPECT_NAME_COLON,
            SchemaState.EXPECT_NAME_COLON: SchemaState.EXPECT_FUNCTION_NAME,
            SchemaState.EXPECT_FUNCTION_NAME: SchemaState.EXPECT_NAME_COMMA,
            SchemaState.EXPECT_NAME_COMMA: SchemaState.EXPECT_PARAMETERS_KEY,
            SchemaState.EXPECT_PARAMETERS_KEY: SchemaState.EXPECT_PARAMETERS_COLON,
            SchemaState.EXPECT_PARAMETERS_COLON: SchemaState.EXPECT_PARAMETERS_OPEN,
            SchemaState.EXPECT_PARAMETERS_OPEN: SchemaState.EXPECT_PARAMETER_NAME,
            SchemaState.EXPECT_PARAMETER_NAME: SchemaState.EXPECT_PARAMETER_COLON,
            SchemaState.EXPECT_PARAMETER_COLON: SchemaState.EXPECT_PARAMETER_VALUE,
            SchemaState.EXPECT_PARAMETER_VALUE: SchemaState.EXPECT_PARAMETER_OR_END,
            SchemaState.EXPECT_OBJECT_END: SchemaState.DONE,
        }

        if self.schema.state in transitions:
            self.schema.state = transitions[self.schema.state]

        if self.schema.state == SchemaState.DONE:
            self.schema.finished = True