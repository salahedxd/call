import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .prompt_validation import prompt_validator
from .functions_validation import functions_validator
from .constrained_decoder import ConstrainedDecoder

# adapter is a methode uses namespace to convert list of dictionary 
# tht contains functions info to an object tht easily be acessed
# SimpleNamespace is a tool for grouping values under named attributes.

def adapter(function):
    parameters = {}
    for name, data in function["parameters"].items():
        parameters[name] = SimpleNamespace(type=data["type"])
    return SimpleNamespace(
        name=function["name"],
        description=function["description"],
        parameters=parameters
    )

def convert_number_parameters(result, functions):
    function = functions[result["name"]]

    for name, parameter in function.parameters.items():
        if parameter.type == "number":
            value = result["parameters"].get(name)

            if isinstance(value, int):
                result["parameters"][name] = float(value)

    return result

def main():

    # argparse is a Python standard-library module.
    # It is a toolbox containing pre-written code such as classes,
    # functions, and methods for working with command-line arguments.

    # ArgumentParser is a class provided by argparse.
    # It is used to define and process the arguments that our program
    # can receive from the command line.

    # Here we create an instance (object) of the ArgumentParser class.
    parser = argparse.ArgumentParser()


    # add_argument() is a method of the parser object.
    # We use it to define an argument that the user can provide
    # from the command line.
    # ruuuuuuuuule book
    # "--functions_definition" is the name/flag of the argument.
    # If the user provides it, its value will be stored.
    # If the user does not provide it, the default value is used.

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


    # parse_args() reads the arguments provided through the CLI.
    # It creates a Namespace object containing the arguments as attributes.
    #
    # Namespace(
    #     input="tests.json",
    #     output="result.json",
    #     functions_definition="data/input/functions_definition.json"
    # )
    #
    # Each attribute contains either the value provided by the user
    # or the corresponding default value.
    # parse_args(), so unknown arguments are rejected and raise error.
    # That's actually useful because it prevents someone
    # from accidentally typing an unsupported option.

    args = parser.parse_args()
    # here we pass the prompts to the validator and store them in requets after validation 
    # and returns a valid list of dictionaries contains prompt as a key and prompts itself as a value.
    try:
        requests = prompt_validator(args.input)

        raw_functions = functions_validator(
            args.functions_definition
        )

    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"Error: {e}")
        return

    # make a empty dictionary called functions
    functions = {}

    # for each function we take it as a key and call adapter for tht function 
    # to create an object that will be the value of the key
    # so we can do functions["mame_fun"].name or function["mame_fun"].parametres["b"].type
    # It's a design choice to give the rest of the program a cleaner, structured interface.

    for function in raw_functions:
        functions[function["name"]] = adapter(function)

    # we create a constrained decoder object from the functions
    decoder = ConstrainedDecoder(functions)

    # creating an empty list to store the result of each prompt.
    results = []

    # Loop over the prompts and store the value in prompt variable
    for request in requests:
        prompt = request["prompt"]

        try:
            # Try to generate the function call
            result = decoder.decode(prompt)

            # till this moment result is still a string.
            # so json.loads() takes a JSON string and converts it into a Python object.

            parsed_result = json.loads(result)
            parsed_result = convert_number_parameters(parsed_result, functions)
            results.append(parsed_result)
        except Exception as e:
            # showing warning messgae during the process 
            print(f"Warning ... {prompt}: {e}")
            # keep processing instead of wuitting the program
            results.append({
                "prompt": prompt,
                "name": "fn_not_found",
                "parameters": {"a": None},
            })

            # *************************************************************

    Path("data/output").mkdir(parents=True, exist_ok=True)

    # The / here is path joining, not mathematical division.
    output_path = Path("data/output") / args.output

    # parents = true means create the whole directory if not exist or create one if missed
    # exist_ok=True This means If the directory already exists, don't give me an error.
    # FileExistError

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # create the file with open if not exist and if its overwrite on it
    # use ascii to read all the caracters on it

    with output_path.open("w", encoding="utf-8") as file:
        # write the results on a file and be good formatted with intent 4 instead of being as one line 
        json.dump(results, file, indent=4)

    print(f"Wrote {len(results)} results to {output_path}")


if __name__ == "__main__":

    try:
        main()

    # KeyboardInterrupt and Exception are both directly under BaseException:
    except KeyboardInterrupt:
        print("User stopped")
    except Exception as e:
        print(f"Error: {e}")
