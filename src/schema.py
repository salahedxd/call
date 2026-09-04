import json
import math
import re

# Characters that legitimately extend a just-closed bracket/paren construct
# (e.g. the "+" in "[0-9]+"). Used to avoid cutting a regex atom off right
# before its quantifier.
# QUANTIFIER_LEAD_CHARS = frozenset("*+?{")


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

    # Schema is the rule engine that knows what the next
    # valid token is at every point of the function-call JSON.

    # allowed_ids = self.schema.allowed_tokens()
    # "Which tokens am I allowed to generate right now?"
    # Then:
    # self.schema.consume(token_id)
    # "I generated this token. Update your state."

    def __init__(self, functions, model, prompt):
        self.functions = functions
        self.model = model
        self.prompt = prompt

        # Substrings the prompt itself sets off in quotes, e.g. for
        # "...the word 'cat' with 'dog' in 'The cat sat...'" this is
        # {"cat", "dog", "The cat sat..."}. A synthesized value that
        # exactly completes one of these spans has very likely finished
        # its job, even if a *longer* literal match against the prompt
        # would also be technically possible (e.g. "cat" vs "cat sat on
        # the mat..." are both substrings of the prompt, but only "cat"
        # is a complete quoted span). This never looks at what the spans
        # actually say - just where the prompt itself drew boundaries.
        self.quoted_spans = self._extract_quoted_spans(prompt)

        self.state = SchemaState.START
        self.selected_function = None
        self.current_parameter = None

        # expected ids its the storage for encode return .. tht contains all the ids of a sequence
        self.expected_ids = []
        self.expected_index = 0

        self.function_sequences = []
        self.active_sequences = []
        self.sequence_index = 0

        self.parameter_sequences = []
        self.active_parameter_sequences = []
        self.parameter_sequence_index = 0
        self.used_parameters = set()

        self.value_type = None
        self.value_buffer = ""
        self.value_started = False
        self.value_token_count = 0

        self.finished = False

        self.max_string_tokens = 20
        self.safe_string_tokens = None

        # Used while the value being generated is still an exact substring
        # of the prompt (e.g. source_string copying text verbatim). The
        # model tends to be genuinely confident about closing in that
        # case, so we can afford to trust it.
        self.quote_confidence_threshold = 0.15  # tune this against real runs

        # Used once the value has diverged from the prompt text - i.e. the
        # model is synthesizing content (a regex, a symbolic replacement)
        # rather than copying it. A small model is rarely >15% confident
        # about closing after a single invented symbol, so for invented
        # content we accept much weaker confidence as "good enough to
        # close" instead of letting it run on.
        self.synthesis_quote_confidence_threshold = 0.02  # tune this too

    def select_token(self, logits, allowed_ids):

        is_string_value = (
            self.state == SchemaState.EXPECT_PARAMETER_VALUE
            and self.value_type == "string"
            and self.value_started
        )

        if not is_string_value:
            return max(allowed_ids, key=lambda t: logits[t])
        # **************************************************************
        # That special block is used when all three are true:

        # We are currently generating a parameter value
        # The parameter's type is string
        # We have already started generating the string

        quote_id = self.encode('"')[0]
        content_ids = []

        for token_id in allowed_ids:
            if token_id != quote_id:
                content_ids.append(token_id)

        # If the quote is the only legal token (e.g. we've hit the
        # max_string_tokens safety cap), there's nothing to filter or
        # compare — just close the string.
        if not content_ids:
            return quote_id

        # "Should this string continue, or have we already found the value we need?"
        appeared_in_prompt = self._value_appeared_in_prompt()
        # this check exists because strings can be normal text OR patterns like regexes.
        has_open_construct = "[" in self.value_buffer or "(" in self.value_buffer

        if (
            not appeared_in_prompt
            and not has_open_construct
            # We've already generated at least one token for this string. We don't want to close an empty string immediately.
            and self.value_token_count > 0
            # The Schema currently allows us to generate the closing ".So we actually have the option to finish the string.
            and quote_id in allowed_ids
        ):
            # Choose " as the next token.
            return quote_id

        if (
            # We have generated something — it's not empty.
            self.value_buffer
            # quoted_spans contains pieces of text that were inside quotes in the user's prompt.
            and self.value_buffer in self.quoted_spans
            and not has_open_construct
            and quote_id in allowed_ids
        ):
            return quote_id

        # This will contain tokens that we decide are safe to use.
        safe_content_ids = []
        # It stores tokens that were rejected by this particular rule:
        blocked_prompt_tokens = []

        # "Look at every possible content token and reject the ones that could create a bad string."
        for token_id in content_ids:
            text = self.model.decode([token_id])
            # if not text:
            #     continue
            if text.lstrip().startswith("\\"):
                continue
            # "What would my string become if I choose this token?"
            prospective = self.value_buffer + text

            if (
                # # we already have content
                # self.value_buffer
                # # The content consists of letters/numbers.
                self.value_buffer.isalnum()
                # Does the new token start with one of these regex symbols?
                and text[0] in ".*+?\\"
            ):
                continue

            if (
                # We haven't generated anything for this string yet.
                self.value_token_count == 0
                # lstrip() removes spaces from the beginning.
                # Does this token start with (?
                and text.lstrip().startswith("(")
                # Would this new value appear in the user's prompt?
                and prospective not in self.prompt
            ):
                continue

            if (
                self.value_buffer
                and appeared_in_prompt
                # If we add the new token, the resulting value would no longer appear in the prompt.
                and prospective not in self.prompt
                # and not has_open_construct
                and not text.lstrip().startswith("[")
            ):
                blocked_prompt_tokens.append(token_id)
                continue

            # “This token passed all our safety checks, so add it to the list of tokens we're allowed to choose from.”
            safe_content_ids.append(token_id)

        # empty list
        if not safe_content_ids:
            # It checks three things:

            # blocked_prompt_tokens
            # → We had tokens that were rejected by the prompt-matching rule.
            # quote_id in allowed_ids
            # → We are legally allowed to close the string with ".
            # self.value_token_count > 0
            # → The string already contains at least one token, so we don't close an empty string.
            if (
                blocked_prompt_tokens
                and quote_id in allowed_ids
                and self.value_token_count > 0
            ):
                # We couldn't find a safe content token, so close the string instead
                return quote_id

            # fallback "Okay, our safety filters rejected everything. Don't get stuck—use the original allowed content tokens."
            safe_content_ids = content_ids

        # "Among the tokens we're allowing, which one has the highest LLM score?"
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
                if token_id == quote_id:
                    continue

                score = logits[token_id]

                if score > best_other_score:
                    best_other_score = score

            if quote_score >= best_other_score:
                return quote_id

        return best_content

    def _has_balanced_brackets(self):
        buf = self.value_buffer

        if "[" not in buf and "(" not in buf:
            return False

        depth = 0

        for ch in buf:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
        # Check whether something is still open
        if depth != 0:
            return False

        return True

    def _value_appeared_in_prompt(self):
        return self.value_buffer in self.prompt

    @staticmethod
    def _extract_quoted_spans(prompt):
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

    def _should_extend_regex(self, candidate_id):
        text = self.model.decode([candidate_id])
        quantifiers = ["*", "+", "?", "{"]

        return text in quantifiers

    def encode(self, text):
        return self.model.encode(text).squeeze(0).tolist()

    def allowed_tokens(self):

        if self.state == SchemaState.EXPECT_FUNCTION_NAME:
            if not self.active_sequences:
                self.load_function_sequences()
            return self.allowed_function_tokens()

        if self.state == SchemaState.EXPECT_PARAMETER_NAME:
            if not self.active_parameter_sequences:
                self.load_parameter_sequences()
            return self.allowed_parameter_tokens()

        if self.state == SchemaState.EXPECT_PARAMETER_VALUE:
            return self.allowed_value_tokens()

        if self.state == SchemaState.EXPECT_PARAMETER_OR_END:
            return self.allowed_parameter_end_tokens()

        if not self.expected_ids:
            self.load_expected()

        return [self.expected_ids[self.expected_index]]

    def load_expected(self):
        expected = None

        if self.state == SchemaState.START:
            expected = "{"
        elif self.state == SchemaState.EXPECT_PROMPT_KEY:
            expected = '"prompt"'
        elif self.state == SchemaState.EXPECT_PROMPT_COLON:
            expected = ":"

        # We use json.dumps() to convert the user's prompt into a 
        # JSON-formatted string before encoding it into tokens.”
        # “Because the prompt will be part of the generated JSON object,
        # so it must follow JSON string formatting,
        # including quotation marks and escaping special characters.”
    
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

            # should raise if no one treated

        if expected is not None:
            self.expected_ids = self.encode(expected)
            # encode() returns a list of token IDs. 0 means the first one
            self.expected_index = 0

    def load_function_sequences(self):

        # A list containing tuples,
        # where each tuple contains a string and a list of token IDs.

        self.function_sequences = []

        # Encode every function name
        for name in self.functions:
            tokens = self.encode(f'"{name}"')
            self.function_sequences.append((name, tokens))

            # [
            #     ("fn_add_numbers", [10, 20, 30]),
            #     ("fn_greet", [10, 20, 40]),
            # ]

        # Copy them into active sequences
        self.active_sequences = self.function_sequences.copy()
        self.sequence_index = 0

    def allowed_function_tokens(self):
        allowed = []

        for name, sequence in self.active_sequences:
            if self.sequence_index < len(sequence):
                # storing the matched token id for each function name
                token_id = sequence[self.sequence_index]

                # fn_add_numbers → [10, 20, 30]
                #                 ↑
                #                 10

                # fn_greet       → [10, 20, 40]
                #                 ↑
                #                 10

                # fn_reverse     → [10, 20, 50]
                #                 ↑
                #                 10

                allowed.append(token_id)

        return allowed

    def load_parameter_sequences(self):
        if self.selected_function is None:
            raise ValueError("No function selected")

        # we store the function object in the local variable function
            # name="fn_add_numbers",
            # description="Add two numbers",
            # parameters={
            #     "a": ...,
            #     "b": ...
            # }

        function = self.functions[self.selected_function]

        self.parameter_sequences = []

        # {
        #     "a": ...,
        #     "b": ...
        # }

        # We go through each parameter name.

        for name in function.parameters:
            if name not in self.used_parameters:
                tokens = self.encode(f'"{name}"')
                self.parameter_sequences.append((name, tokens))

        self.active_parameter_sequences = self.parameter_sequences.copy()
        self.parameter_sequence_index = 0

    def allowed_parameter_tokens(self):
        allowed = []

        for name, sequence in self.active_parameter_sequences:
            if self.parameter_sequence_index < len(sequence):
                token_id = sequence[self.parameter_sequence_index]
                allowed.append(token_id)

        return allowed

    def load_parameter_value(self):
        function = self.functions[self.selected_function]
        parameter = function.parameters[self.current_parameter]

        self.value_type = parameter.type
        self.value_buffer = ""
        self.value_token_count = 0
        self.value_started = False

    def allowed_value_tokens(self):

        if self.value_type is None:
            self.load_parameter_value()

        if self.value_type == "boolean":
            return self.encode("true") + self.encode("false")

        if self.value_type == "null":
            return self.encode("null")

        if self.value_type in ("number", "integer"):
            prompt_tokens = self.get_prompt_number_tokens()

            if prompt_tokens:
                if self.value_token_count < len(prompt_tokens):
                    return [prompt_tokens[self.value_token_count]]

                return self.allowed_parameter_end_tokens()

            return self.get_number_tokens(self.value_type)

        if self.value_type == "string":
            quote_id = self.encode('"')[0]
            escaped_quote_id = self.encode('\\"')[0]

            if not self.value_started:
                return [quote_id]

            if self.value_token_count == self.max_string_tokens:
                return [quote_id]

            content_tokens = self.get_string_content_tokens()

            if self.value_token_count == 0:
                # A value must contain at least one character before it's
                # allowed to close — an immediately-empty string is never
                # the intended answer for a value the model was asked to
                # produce.
                return content_tokens + [escaped_quote_id]

            return content_tokens + [escaped_quote_id, quote_id]

        raise ValueError(
            f"Unsupported parameter type: {self.value_type}"
        )

    def get_prompt_number_tokens(self):
        """get_prompt_number_tokens() finds the numbers inside 
        the user's prompt and converts them into token IDs."""
        numbers = []

        for word in self.prompt.split():
            word = word.strip(".,!?")
            
            try:
                float(word)
                numbers.append(word)
            except ValueError:
                continue

        if not numbers:
            return []

        function = self.functions[self.selected_function]
        parameter_names = list(function.parameters)

        parameter_index = parameter_names.index(self.current_parameter)

        return self.encode(numbers[parameter_index])

    def get_number_tokens(self, value_type):
        allowed = []

        for token_id in range(self.model._tokenizer.vocab_size):
            token_text = self.model.decode([token_id])

            if not token_text:
                continue

            candidate = self.value_buffer + token_text

            if candidate == "-":
                allowed.append(token_id)
                continue

            digits = candidate[1:] if candidate.startswith("-") else candidate

            if digits.isdigit():
                allowed.append(token_id)
                continue

            if value_type == "number":
                if (
                    digits.count(".") == 1
                    and digits.replace(".", "").isdigit()
                ):
                    allowed.append(token_id)

        return allowed

    def allowed_parameter_end_tokens(self):
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

    def get_string_content_tokens(self):

        # make a safe string tokens if not exist
        if self.safe_string_tokens is not None:
            return self.safe_string_tokens

        # The whitelist is basically:
        # “These are the tokens the LLM is allowed to use while we're inside a string.”
        # And we exclude tokens containing an unescaped 
        # ", because a raw " would be interpreted as the end of the JSON string.

        quote_id = self.encode('"')[0]
        allowed = []

        for token_id in range(self.model._tokenizer.vocab_size):
            if token_id == quote_id:
                continue

            token_text = self.model.decode([token_id])

            if not token_text:
                continue

            if '"' in token_text and '\\"' not in token_text:
                continue

            # A raw backslash is only valid JSON if the very next
            # character is a recognized escape (\\, \", \/, \b, \f, \n,
            # \r, \t, or \u....). We generate one token at a time and
            # can't guarantee that pairing, so any token containing a
            # backslash that isn't one of those exact known-safe
            # sequences is excluded outright. In practice this just
            # nudges regex synthesis toward bracket notation ("[0-9]")
            # instead of backslash shorthand ("\d"), which keeps every
            # generated string valid JSON without needing a repair pass.
            if "\\" in token_text and token_text not in (
                "\\\\", "\\\"", "\\/", "\\b", "\\f", "\\n", "\\r", "\\t",
            ):
                continue

            # Raw control characters (unescaped newlines, tabs, etc.)
            # are never valid inside a JSON string literal.
            if any(ord(ch) < 0x20 for ch in token_text):
                continue

            allowed.append(token_id)

        self.safe_string_tokens = allowed
        return self.safe_string_tokens