from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from .schema import Schema
from .consumer import SchemaConsumer


class ConstrainedDecoder:
    """Generate function calls while enforcing Schema constraints."""

    def __init__(self, functions: dict) -> None:

        self.model = Small_LLM_Model()
        self.functions = functions
        self.max_steps = 150

    def decode(self, prompt: str) -> str:

        schema = Schema(
            self.functions,
            self.model,
            prompt,
        )

        consumer = SchemaConsumer(schema)

        catalog = ""
        for name, function in self.functions.items():
            catalog += f"- {name}: {function.description}\n"

        model_prompt = (
            f"Available functions:\n{catalog}\n\n"
            f"User request: {prompt}\n\n"
            "Generate the function call JSON."
        )

        context_ids = (
            self.model.encode(model_prompt)
            .squeeze(0)
            .tolist()
        )

        result = []

        for _ in range(self.max_steps):

            if schema.finished:
                break

            allowed_ids = schema.allowed_tokens()

            if not allowed_ids:
                raise ValueError(
                    f"No valid tokens available in schema state "
                    f"{schema.state}"
                )

            if len(allowed_ids) == 1:
                token_id = allowed_ids[0]
            else:
                logits = self.model.get_logits_from_input_ids(context_ids)
                token_id = schema.select_token(logits, allowed_ids)

            result.append(token_id)
            consumer.consume(token_id)
            context_ids.append(token_id)

        if not schema.finished:
            raise RuntimeError(
                f"Constrained generation exceeded {self.max_steps} steps"
            )

        return str(self.model.decode(result))
