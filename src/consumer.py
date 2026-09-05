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
        # . Its main job is to use the generated token to eliminate
        # function candidates that no longer match.
        schema = self.schema
        valid_sequences = []
        # we loop over the twin name and sequence of the
        # active function sequences
        for name, sequence in schema.active_sequences:
            # the condition is basically a guard to ensure we don't go out of bounds when checking the sequence
            if schema.sequence_index < len(sequence):
                if sequence[schema.sequence_index] == token_id:
                    valid_sequences.append((name, sequence))

        schema.active_sequences = valid_sequences

        # if not schema.active_sequences:
        #     raise ValueError("Invalid function name token")

        schema.sequence_index += 1

        completed = []

        for name, sequence in schema.active_sequences:
            if schema.sequence_index == len(sequence):
                completed.append((name, sequence))

        if completed:
            # here we take the first completed function name
            # and set it as the selected function [0]
            # its the binary name sequensce while [0][0] means the name of the function
            schema.selected_function = completed[0][0]
            schema.function_sequences = []
            schema.active_sequences = []
            schema.sequence_index = 0
            self.advance_state()

    def consume_parameter_token(self, token_id):
        schema = self.schema
        valid_sequences = []

        for name, sequence in schema.active_parameter_sequences:
            # a guard to not depass te length of the sequence when checking the current token
            if schema.parameter_sequence_index < len(sequence):
                # filter the active parameter sequences to only keep those that match the current token
                if sequence[schema.parameter_sequence_index] == token_id:
                    valid_sequences.append((name, sequence))
        # update the active parameter sequences to only include those that matched the current token
        schema.active_parameter_sequences = valid_sequences
        # advance the parameter sequence index to move to the next token in the sequence
        schema.parameter_sequence_index += 1

        completed = []
        # check if any of the active parameter sequences have been fully
        # matched and select the first completed parameter name as the current parameter
        for name, sequence in schema.active_parameter_sequences:
            if schema.parameter_sequence_index == len(sequence):
                completed.append((name, sequence))

        if completed:
            # here we take the first completed parameter name
            # and set it as the current parameter
            schema.current_parameter = completed[0][0]
            # we add the current parameter to the set of used parameters to avoid reusing it
            schema.used_parameters.add(schema.current_parameter)
            # # we advance the parameter index to move to the next parameter in the function's parameter list
            # schema.parameter_index += 1
            # we reset the active parameter sequences and active parameter sequence index to prepare for the next parameter
            # and the parameter sequence index
            schema.parameter_sequences = []
            schema.active_parameter_sequences = []
            schema.parameter_sequence_index = 0

            # we advance the state to expect the parameter colon next
            self.advance_state()

    def consume_parameter_end(self, token_id):
        schema = self.schema

        comma_id = schema.encode(",")[0]
        close_id = schema.encode("}")[0]
        # reset the value-related attributes to prepare for the next parameter
        # or to finalize the function call
        schema.value_type = None
        schema.value_buffer = ""
        schema.value_token_count = 0

        # we check if the token is a comma or a closing brace to determine
        # the next state
        if token_id == comma_id:
            schema.parameter_index += 1
            schema.state = SchemaState.EXPECT_PARAMETER_NAME
            return

        if token_id == close_id:
            schema.state = SchemaState.EXPECT_OBJECT_END
            return

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

            if token_id == comma_id or token_id == close_id:
                if not schema.value_buffer:
                    raise ValueError("Number cannot be empty")

                self.consume_parameter_end(token_id)
                return

            # value_buffer
            # → WHAT have we generated?

            # value_token_count
            # → HOW MANY tokens have we generated?
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
