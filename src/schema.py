import json
import math
import re

# Characters that legitimately extend a just-closed bracket/paren construct
# (e.g. the "+" in "[0-9]+"). Used to avoid cutting a regex atom off right
# before its quantifier.
QUANTIFIER_LEAD_CHARS = frozenset("*+?{")


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
        if not (
            self.state == SchemaState.EXPECT_PARAMETER_VALUE
            and self.value_type == "string"
            and self.value_started
        ):
            return max(allowed_ids, key=lambda t: logits[t])

        quote_id = self.encode('"')[0]
        content_ids = [t for t in allowed_ids if t != quote_id]

        # If the quote is the only legal token (e.g. we've hit the
        # max_string_tokens safety cap), there's nothing to filter or
        # compare — just close the string.
        if not content_ids:
            return quote_id

        is_verbatim = self._is_verbatim_so_far()
        has_open_construct = "[" in self.value_buffer or "(" in self.value_buffer

        # Hard rule, not a probability nudge: once we've left verbatim
        # territory (the value no longer matches a substring of the
        # prompt) without ever opening a bracket construct, there is
        # nothing legitimate left to add. Close now - regardless of how
        # confident the model is about continuing. A soft threshold can
        # always be overridden by a model that's simply very sure about
        # an unwanted continuation; this can't be.
        if (
            not is_verbatim
            and not has_open_construct
            and self.value_token_count > 0
            and quote_id in allowed_ids
        ):
            return quote_id

        # Second hard rule: the buffer exactly completes one of the
        # prompt's own quoted spans (e.g. buffer == "cat" and the prompt
        # contains 'cat'). That's a strong, structural signal that this
        # value is done, even though a *longer* substring of the prompt
        # (e.g. "cat sat on the mat...") might also still match - only a
        # complete quoted span gets this treatment, so it can't fire on
        # an arbitrary partial match.
        if (
            self.value_buffer
            and self.value_buffer in self.quoted_spans
            and not has_open_construct
            and quote_id in allowed_ids
        ):
            return quote_id

        safe_content_ids = []
        blocked_by_verbatim_guard = []

        for token_id in content_ids:
            text = self.model.decode([token_id])
            if not text:
                continue

            prospective = self.value_buffer + text

            if (
                self.value_buffer
                and self.value_buffer.isalnum()
                and text[0] in ".*+?\\"
            ):
                continue

            # Don't let a synthesized value (one not copied verbatim from
            # the prompt) open with a capturing group. "(" as the very
            # first character only ever serves to wrap the whole value in
            # a redundant group (e.g. "([0-9]+)" instead of "[0-9]+") -
            # there's nothing before it that a group could usefully
            # separate. A verbatim copy is left alone in case the quoted
            # source text itself happens to start with "(".
            if (
                self.value_token_count == 0
                and text.lstrip().startswith("(")
                and prospective not in self.prompt
            ):
                continue

            # While the value is still an exact copy of prompt text,
            # don't let a candidate abandon that match unless it's
            # extending the copy further or opening a bracket construct.
            # Anything else (an anchor, a stray symbol, an unrelated
            # word) is unwanted elaboration once a literal match is
            # already sitting there complete - e.g. this is what stops
            # "cat" from growing into "cat$|dog$|cat".
            if (
                self.value_buffer
                and is_verbatim
                and prospective not in self.prompt
                and not has_open_construct
                and not text.lstrip().startswith("[")
            ):
                blocked_by_verbatim_guard.append(token_id)
                continue

            if prospective not in self.prompt and self._would_repeat_adjacent(prospective):
                continue

            safe_content_ids.append(token_id)

        if not safe_content_ids:
            if (
                blocked_by_verbatim_guard
                and quote_id in allowed_ids
                and self.value_token_count > 0
            ):
                # Every remaining candidate would abandon a complete
                # literal match for no good reason - stop instead of
                # picking one anyway.
                return quote_id
            # Otherwise this is just the existing filters being too
            # aggressive for this step - fall back rather than raising.
            safe_content_ids = content_ids

        best_content = max(safe_content_ids, key=lambda t: logits[t])

        if quote_id in allowed_ids and self.value_token_count > 0:
            structurally_complete = self._value_is_structurally_complete()

            # Once a bracket/paren construct has balanced, the decision
            # is made structurally, not probabilistically - either the
            # model is reaching for a quantifier (take it, unconditionally)
            # or it isn't (close, unconditionally). Leaving this to the
            # probability check would let an extreme swing in confidence
            # either strand a needed quantifier or keep a finished
            # construct open for no reason.
            if structurally_complete:
                if self._wants_quantifier_extension(best_content):
                    return best_content
                return quote_id

            max_score = max(logits[t] for t in allowed_ids)
            exps = {t: math.exp(logits[t] - max_score) for t in allowed_ids}
            total = sum(exps.values())
            quote_prob = exps[quote_id] / total

            threshold = (
                self.quote_confidence_threshold
                if is_verbatim
                else self.synthesis_quote_confidence_threshold
            )

            if quote_prob >= threshold:
                return quote_id

        return best_content

    def _would_repeat_adjacent(self, candidate_buffer, max_k=6):
        """Flags back-to-back repetition only — e.g. '*' immediately
        followed by another '*', or 'cat' immediately followed by
        'cat' again. Unlike a global substring search, this ignores
        characters that simply recur later in ordinary text (like the
        two 'm's in "Programming"), so it won't misfire on real words."""
        n = len(candidate_buffer)
        for k in range(1, min(max_k, n // 2) + 1):
            tail = candidate_buffer[-k:]
            prev = candidate_buffer[-2 * k:-k]
            if tail and tail == prev:
                return True
        return False

    def _value_is_structurally_complete(self):
        """Generic bracket-balance check: only relevant once the value has
        opened at least one '(' or '[' construct. It doesn't know anything
        about regex semantics or content — it only tracks paired-symbol
        balance, same as validating any bracketed expression."""
        buf = self.value_buffer
        if "[" not in buf and "(" not in buf:
            return False  # never engages for plain literal strings

        depth = 0
        for ch in buf:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1

        if depth != 0:
            return False

        return buf[-1] != "\\"

    def _is_verbatim_so_far(self):
        """True while the value being generated is still an exact
        substring of the user's prompt - i.e. the model is copying text
        out of the request (typical for source_string). False the moment
        the buffer diverges from the prompt, meaning the model has
        started inventing content instead (typical for a synthesized
        regex or a symbolic replacement like '*'). This only ever
        compares against self.prompt - it never checks for specific
        words, so it generalizes to any request."""
        if not self.value_buffer:
            return True
        return self.value_buffer in self.prompt

    @staticmethod
    def _extract_quoted_spans(prompt):
        """Pull out every substring the prompt itself wrapped in quotes
        (single or double). Purely structural - it doesn't know or care
        what any given span says, only that the prompt marked it off as
        a distinct unit."""
        spans = set()
        for quote_char in ("'", '"'):
            pattern = re.escape(quote_char) + r"([^" + re.escape(quote_char) + r"]*)" + re.escape(quote_char)
            for match in re.finditer(pattern, prompt):
                span = match.group(1)
                if span:
                    spans.add(span)
        return spans

    def _wants_quantifier_extension(self, candidate_id):
        """True only if we've just closed a bracket/paren construct and
        the model's best next candidate is a regex quantifier (*, +, ?,
        or the start of a {n,m} form). This is what lets '[0-9]+' keep
        its '+' instead of being force-closed the instant ']' lands,
        while still closing '[aeiouAEIOU]' immediately since a plain
        letter is never a quantifier."""
        if not self.value_buffer or self.value_buffer[-1] not in ")]":
            return False

        text = self.model.decode([candidate_id])
        return bool(text) and text[0] in QUANTIFIER_LEAD_CHARS

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