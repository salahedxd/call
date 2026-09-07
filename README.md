*This project has been created as part of the 42 curriculum by sasliman.*

# Call Me Maybe

## Description

**Call Me Maybe** is a 42 project about **function calling with Large Language Models (LLMs)**.

The goal of the project is to take a natural-language prompt and produce a structured JSON function call containing:

- the original prompt;
- the selected function;
- the arguments extracted from the prompt.

The project uses the small **Qwen3-0.6B** language model through the provided local `llm_sdk`.

Instead of simply asking the language model to generate JSON and hoping that the result is valid, this project implements **constrained decoding**. At every generation step, the decoder determines which tokens are allowed by the expected JSON structure and function schema, then selects the highest-scoring valid token.

The main pipeline is:

    User Prompt
          ↓
    Tokenization
          ↓
    Input IDs
          ↓
    Qwen3-0.6B
          ↓
    Logits
          ↓
    Schema determines allowed tokens
          ↓
    Best allowed token
          ↓
    Consumer updates schema state
          ↓
    Repeat until DONE
          ↓
    Function-call JSON

The project also validates function definitions and prompts using **Pydantic**.

## Features

- Token-by-token constrained decoding.
- JSON structure enforcement.
- Function-name selection constrained to available functions.
- Parameter-name selection constrained to the selected function.
- Parameter type handling for:
  - `number`
  - `integer`
  - `string`
  - `boolean`
  - `null`
- Number extraction from the user's prompt.
- String-content constraints for safer JSON generation.
- Handling of quoted values.
- Handling of regular-expression-like string values.
- `fn_not_found` fallback for prompts that do not match an available function.
- Input validation with Pydantic.
- Flake8 and mypy checks.

## Algorithm Explanation

### 1. Tokenization

The language model does not directly work with complete words or JSON objects. It generates **tokens**.

The input prompt is first encoded into token IDs.

The generated token IDs are then kept as context for the following generation steps.

### 2. Generate logits

For the current context, the language model produces a score called a **logit** for every token in its vocabulary.

Conceptually:

    token A → score
    token B → score
    token C → score
    ...

Normally, the model could select any token from the vocabulary.

For this project, that would not be reliable enough because an arbitrary token could break the JSON structure or violate the function definition.

### 3. Determine the allowed tokens

The `Schema` class represents the current position inside the expected JSON object.

The decoder moves through a state machine such as:

    START
      ↓
    EXPECT_PROMPT_KEY
      ↓
    EXPECT_PROMPT_COLON
      ↓
    EXPECT_PROMPT_VALUE
      ↓
    EXPECT_NAME_KEY
      ↓
    EXPECT_FUNCTION_NAME
      ↓
    EXPECT_NAME_COMMA
      ↓
    EXPECT_PARAMETERS_KEY
      ↓
    EXPECT_PARAMETERS_COLON
      ↓
    EXPECT_PARAMETERS_OPEN
      ↓
    EXPECT_PARAMETER_NAME
      ↓
    EXPECT_PARAMETER_COLON
      ↓
    EXPECT_PARAMETER_VALUE
      ↓
    EXPECT_PARAMETER_OR_END
      ↓
    EXPECT_OBJECT_END
      ↓
    DONE

At each state, the schema knows what tokens are legal.

For fixed JSON elements such as `{`, `:`, `,`, and `}`, the expected token sequence is known directly.

For dynamic elements such as function names and parameter names, candidate token sequences are generated from the available definitions.

### 4. Function-name selection

When the decoder reaches `EXPECT_FUNCTION_NAME`, it encodes all available function names.

For example:

    "fn_add_numbers"
    "fn_greet"
    "fn_reverse_string"

Only tokens that belong to at least one valid function-name sequence are allowed.

As tokens are generated, `SchemaConsumer` removes candidates that no longer match the generated sequence.

When a complete candidate sequence is generated, that function becomes the selected function.

### 5. Parameter-name selection

After selecting a function, the decoder applies the same idea to its parameters.

Only parameters belonging to the selected function and not already used are considered.

This allows the model to choose the parameter order while preventing invalid parameter names.

### 6. Parameter-value constraints

The allowed tokens depend on the parameter type defined by the selected function.

#### Numbers

For `number` and `integer` parameters, the implementation extracts numbers from the user's prompt.

For example:

    What is the sum of 265 and 345?

The decoder extracts:

    265
    345

and uses the corresponding token sequence for the current parameter.

This prevents the model from freely inventing numerical values.

#### Boolean

Only:

    true
    false

are accepted.

#### Null

Only:

    null

is accepted.

#### Strings

String generation requires more flexibility.

The implementation:

- starts and ends JSON strings with quotation marks;
- limits the number of generated string tokens;
- caches safe string-content token IDs;
- checks generated content against the prompt where appropriate;
- detects quoted spans from the prompt;
- checks bracket and parenthesis balance;
- handles regular-expression quantifiers when required.

### 7. Selecting the best valid token

The decoder does not simply select the highest-scoring token from the entire vocabulary.

Instead:

    All vocabulary tokens
            ↓
    Apply schema constraints
            ↓
    Allowed tokens
            ↓
    Select highest-scoring token

This is the central idea behind the constrained decoder.

The project subject explicitly requires the decoder to enforce both JSON validity and schema compliance instead of relying on the model to spontaneously produce correct JSON. 

### 8. Consuming the selected token

After a token is selected, `SchemaConsumer.consume()` updates the current schema state.

The consumer handles:

- function-name candidate progression;
- parameter-name candidate progression;
- parameter values;
- commas;
- object closing;
- state transitions.

The selected token is then added to the context and the model produces logits for the next token.

This process repeats until the schema reaches `DONE`.

## Design Decisions

### Schema and Consumer separation

The `Schema` class is responsible for determining **what tokens are allowed**.

The `SchemaConsumer` is responsible for handling **the token that was actually generated** and updating the state.

This separation keeps token constraints and state transitions independent.

### State-machine design

A state machine was chosen because the output has a predictable JSON structure.

Each state represents a specific position in the function-call object, making it possible to determine the legal next tokens.

### Dynamic token sequences

Function names and parameter names are not hard-coded into the decoder.

They are converted into token sequences and progressively filtered as tokens are generated.

This makes the decoder reusable with different function definitions.

### Prompt numbers are reused

For numerical parameters, values are extracted from the prompt rather than freely generated by the language model.

This improves reliability for prompts such as:

    What is the sum of 265 and 345?

### Pydantic validation

Pydantic is used to validate structured input data before decoding.

Function definitions are checked for:

- valid function names;
- descriptions;
- parameter definitions;
- parameter types;
- return definitions.

Unexpected fields are rejected.

Prompts are also validated through a Pydantic model.

### Local model

The project uses the provided local `llm_sdk` and Qwen3-0.6B rather than an external API.

This keeps inference local and follows the project's intended environment.

## Instructions

### Requirements

The project requires:

- Python 3.10 or newer;
- `uv`;
- the provided `llm_sdk`;
- the model files required by the SDK.

### Installation

From the project root:

    uv sync

### Running the project

The default command is:

    uv run python -m src

The program also accepts custom input and output files:

    uv run python -m src \
        --functions_definition data/input/functions_definition.json \
        --input data/input/function_calling_tests.json \
        --output data/output/function_calling_results.json

The generated results are written to:

    data/output/function_calling_results.json

### Linting

Run:

    make lint

This runs:

    flake8 src

and:

    mypy src --warn-return-any --warn-unused-ignores \
        --ignore-missing-imports --disallow-untyped-defs \
        --check-untyped-defs

## Example Usage

### Example prompt

    What is the sum of 2 and 3?

If the available functions contain `fn_add_numbers`, the decoder produces a function call similar to:

    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2.0,
            "b": 3.0
        }
    }

Another example:

    Greet shrek

can produce:

    {
        "prompt": "Greet shrek",
        "name": "fn_greet",
        "parameters": {
            "name": "shrek"
        }
    }

The output is stored as JSON in the configured output file.

## Testing Strategy

Testing was performed at several levels.

### Static analysis

The source code is checked with:

    make lint

This runs both Flake8 and mypy.

### Input validation

Function definitions and prompts are validated using Pydantic before the decoding process starts.

Malformed definitions are rejected instead of being passed directly to the decoder.

### Function-calling tests

The provided function-calling test set is executed and the resulting JSON file is inspected.

The results are checked for:

- valid JSON;
- correct function selection;
- correct parameter names;
- correct parameter values;
- correct parameter types.

### Edge cases

The implementation was tested with cases involving:

- multiple parameters;
- numbers;
- strings;
- quoted strings;
- regular-expression-like values;
- boolean values;
- null values;
- prompts that do not match a function;
- special string content.

The project subject also recommends testing empty strings, large numbers, special characters, wrong types, ambiguous prompts, and functions with multiple parameters.

## Performance Analysis

The project requirements specify:

- at least **90% accuracy** for function selection and argument extraction;
- **100% valid JSON**;
- processing the test prompts in **under 5 minutes** on standard hardware;
- robust error handling.

### Function Selection

The current implementation achieves:

    100% function selection accuracy

Every test prompt is mapped to the correct function, including prompts that require selecting the `fn_not_found` fallback.

### Parameter Extraction

Parameter extraction currently achieves **above 90% accuracy** on the tested prompts.

The main remaining errors are related to more difficult string-value extraction cases, while function selection itself is fully reliable on the current test set.

### JSON Reliability

The decoder constrains generation token-by-token rather than asking the model to freely generate a JSON object.

This prevents many structural errors before they can be generated.

The generated structure is controlled by the schema state machine.

### Speed

A complete local run currently takes approximately:

    2 minutes 30 seconds

on the development machine.

This is below the project's target of 5 minutes, so the current implementation meets the required execution-time target.

Performance was improved by reducing unnecessary Python-side processing during token generation, particularly around vocabulary and token handling.

### Reliability

The constrained state machine improves reliability by removing invalid structural tokens before token selection.

This allows the relatively small Qwen3-0.6B model to perform structured function calling without relying entirely on its ability to generate a complete JSON object correctly.

## Challenges Faced

### 1. Reducing Processing Time

One of the main challenges was reducing the processing time of the decoder.

The initial implementation took approximately:

    9 minutes

to process the complete test set, which was above the project's 5-minute requirement.

The main issue was unnecessary processing of the model's vocabulary, especially repeated vocabulary loops during token selection.

The solution was to reduce unnecessary vocabulary processing and avoid looping through tokens when it was not required.

After these optimizations, the processing time was reduced to approximately:

    2 minutes 30 seconds

which is below the project's 5-minute requirement.

### 2. Following the JSON Structure

Another major challenge was making the decoder follow the required JSON structure correctly.

The model generates one token at a time, so the decoder needs to know exactly what can come next at every step.

The solution was to implement a **state machine**.

Each state represents a position in the JSON structure, such as:

    START
    EXPECT_PROMPT_KEY
    EXPECT_PROMPT_VALUE
    EXPECT_FUNCTION_NAME
    EXPECT_PARAMETER_NAME
    EXPECT_PARAMETER_VALUE
    EXPECT_PARAMETER_OR_END
    EXPECT_OBJECT_END
    DONE

The schema uses the current state to determine which tokens are allowed, while the consumer updates the state after each generated token.

This made it possible to control the JSON structure instead of relying on the language model to generate valid JSON by itself.

### 3. String Generation

String values were one of the most challenging parts because their content cannot be constrained as simply as fixed JSON elements such as `{`, `:`, `,`, or `}`.

The solution introduced:

- safe string token filtering;
- quoted-span detection;
- prompt matching;
- bracket and parenthesis balancing;
- regular-expression handling;
- a maximum string token limit.

### 4. Handling Parameter Values Independently

Another challenge was handling parameter values because different parameter types require different rules.

The solution was to handle each parameter type independently according to its schema definition.

For example:

- `number` and `integer` values use numbers extracted from the prompt;
- `boolean` values are restricted to `true` or `false`;
- `null` values are restricted to `null`;
- `string` values use dedicated string-generation constraints.

This allowed the decoder to apply the appropriate constraints depending on the selected parameter instead of treating all values in the same way.

## Project Structure

    CallMeMaybe/
    ├── pyproject.toml
    ├── uv.lock
    ├── Makefile
    ├── README.md
    ├── src/
    │   ├── __main__.py
    │   ├── constrained_decoder.py
    │   ├── schema.py
    │   ├── consumer.py
    │   ├── functions_validation.py
    │   ├── prompt_validation.py
    │   └── vocabulary.py
    ├── llm_sdk/
    │   ├── pyproject.toml
    │   └── llm_sdk/
    │       └── __init__.py
    └── data/
        ├── input/
        │   ├── functions_definition.json
        │   └── function_calling_tests.json
        └── output/
            └── function_calling_results.json

## Resources

### Documentation and references

- **42 Call Me Maybe subject** — the main project specification and requirements.
- **Hugging Face Transformers documentation** — reference for transformer models and inference.
- **Python JSON documentation** — reference for JSON parsing and serialization.
- **Pydantic documentation** — reference for structured data validation.
- **Python typing documentation** — reference for type annotations and typed interfaces.

### AI Usage

AI tools were used as a development and learning assistant during the project.

They were used for:

- explaining LLM concepts such as tokenization, logits, and constrained decoding;
- explaining state machines and token-by-token generation;
- helping understand and resolve mypy and Flake8 errors;
- explaining Pydantic and helping integrate structured validation;
- reviewing implementation logic during debugging;
- helping structure the project documentation and README.

AI was not used as a replacement for understanding or testing the implementation. Suggestions were reviewed, adapted to the project requirements, and tested against the actual implementation.

## Conclusion

Call Me Maybe demonstrates how a small language model can be guided to produce structured function calls through **constrained decoding**.

The central idea is:

    Do not ask the model to produce valid JSON.
    Control which tokens it is allowed to produce.

By combining the model's logits with a schema-driven state machine, the project transforms unconstrained language generation into a controlled function-calling process.