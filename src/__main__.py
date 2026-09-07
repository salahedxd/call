import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from .prompt_validation import prompt_validator
from .functions_validation import functions_validator
from .constrained_decoder import ConstrainedDecoder


def adapter(function: dict) -> SimpleNamespace:
    """Convert a validated function dictionary into a function object."""

    parameters = {}
    for name, data in function["parameters"].items():
        parameters[name] = SimpleNamespace(type=data["type"])
    return SimpleNamespace(
        name=function["name"],
        description=function["description"],
        parameters=parameters
    )


def convert_number_parameters(
    result: dict,
    functions: dict
) -> dict:
    """Convert integer results to floats for number parameters."""

    function = functions[result["name"]]
    for name, parameter in function.parameters.items():
        if parameter.type == "number":
            value = result["parameters"].get(name)

            if isinstance(value, int):
                result["parameters"][name] = float(value)

    return result


def main() -> None:
    """Parse arguments, decode prompts, and save function-calling results."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )

    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )

    parser.add_argument(
        "--output",
        default="function_calling_results.json",
    )

    args = parser.parse_args()

    try:
        requests = prompt_validator(args.input)
        raw_functions = functions_validator(
            args.functions_definition
        )

    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"Error: {e}")
        return

    functions = {}
    for function in raw_functions:
        functions[function["name"]] = adapter(function)

    decoder = ConstrainedDecoder(functions)
    results = []

    for request in requests:
        prompt = request["prompt"]

        try:
            result = decoder.decode(prompt)

            parsed_result = json.loads(result)
            parsed_result = convert_number_parameters(parsed_result, functions)
            results.append(parsed_result)
        except Exception as e:

            print(f"Warning ... {prompt}: {e}")
            results.append({
                "prompt": prompt,
                "name": "fn_not_found",
                "parameters": {"a": None},
            })

    Path("data/output").mkdir(parents=True, exist_ok=True)
    output_path = Path("data/output") / args.output

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"Wrote {len(results)} results to {output_path}")


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("User stopped")
    except Exception as e:
        print(f"Error: {e}")
