import ast
import json


def format_json_like_text(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        raw_data = f.read()

    try:
        # Convert string representation of Python list of dicts into actual data structure
        data = ast.literal_eval(raw_data)
    except Exception as e:
        print("Failed to parse input:", e)
        return

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Formatted output written to {output_path}")


if __name__ == "__main__":
    format_json_like_text("dump.txt", "dump_formatted.txt")
