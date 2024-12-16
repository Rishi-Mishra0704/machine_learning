import os
import argparse
import jupytext

# Define directories
notebook_dir = 'notebook'
script_dir = 'script'

# Function to create directories if they don't exist
def create_dirs():
    # Ensure that both notebook and script directories exist
    os.makedirs(notebook_dir, exist_ok=True)
    os.makedirs(script_dir, exist_ok=True)

# Function to convert .ipynb to .py using jupytext
def convert_ipynb_to_py(ipynb_file):
    input_path = os.path.join(notebook_dir, ipynb_file)
    output_file = os.path.join(script_dir, f"{os.path.splitext(ipynb_file)[0]}.py")
    
    # Load the notebook and save it as a Python script using jupytext
    notebook = jupytext.read(input_path)
    jupytext.write(notebook, output_file)

    print(f"Converted {ipynb_file} to {output_file}")

# Function to convert .py to .ipynb using jupytext (splitting code into multiple cells)
def convert_py_to_ipynb(py_file):
    input_path = os.path.join(script_dir, py_file)
    output_file = os.path.join(notebook_dir, f"{os.path.splitext(py_file)[0]}.ipynb")

    # Load the Python script and save it as a Jupyter notebook using jupytext
    script = jupytext.read(input_path)
    jupytext.write(script, output_file)

    print(f"Converted {py_file} to {output_file}")


# Function to handle the conversion process
def convert_files(to_format):
    create_dirs()  # Create necessary directories before conversion

    if to_format == 'py':
        # Convert all .ipynb files in the notebook directory to .py files in the script directory
        for filename in os.listdir(notebook_dir):
            if filename.endswith('.ipynb'):
                convert_ipynb_to_py(filename)

    elif to_format == 'ipynb':
        # Convert all .py files in the script directory to .ipynb files in the notebook directory
        for filename in os.listdir(script_dir):
            if filename.endswith('.py'):
                convert_py_to_ipynb(filename)

    else:
        print("Invalid format specified. Use 'py' or 'ipynb'.")

    print("Conversion completed!")

# Main function to parse command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Convert between .ipynb and .py files.")
    parser.add_argument('--to', choices=['py', 'ipynb'], required=True, 
                        help="Specify the conversion direction: 'py' for converting notebooks to Python scripts, 'ipynb' for converting Python scripts to notebooks.")
    
    args = parser.parse_args()
    
    # Perform the conversion based on the argument
    convert_files(args.to)

if __name__ == '__main__':
    main()
