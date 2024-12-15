import os
import argparse
import nbformat

# Define directories
notebook_dir = 'notebook'
script_dir = 'script'

# Function to convert .ipynb to .py
def convert_ipynb_to_py(ipynb_file):
    input_path = os.path.join(notebook_dir, ipynb_file)
    output_file = os.path.join(script_dir, f"{os.path.splitext(ipynb_file)[0]}.py")

    # Read the notebook file
    with open(input_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)

    # Extract code cells and write to .py file
    with open(output_file, 'w', encoding='utf-8') as py_file:
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                py_file.write(cell.source + '\n\n')

    print(f"Converted {ipynb_file} to {output_file}")

# Function to convert .py to .ipynb (splitting code into multiple cells)
def convert_py_to_ipynb(py_file):
    input_path = os.path.join(script_dir, py_file)
    output_file = os.path.join(notebook_dir, f"{os.path.splitext(py_file)[0]}.ipynb")

    # Read the Python file
    with open(input_path, 'r', encoding='utf-8') as f:
        py_code = f.read()

    # Split the Python code into multiple blocks (here assuming `# %%` is the cell delimiter)
    code_blocks = py_code.split('\n\n')  # Adjust based on your code structure, e.g., using comments or `# %%`

    # Create a new notebook and add code cells
    notebook = nbformat.v4.new_notebook()
    for block in code_blocks:
        notebook.cells.append(nbformat.v4.new_code_cell(block))

    # Write to the .ipynb file
    with open(output_file, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)

    print(f"Converted {py_file} to {output_file}")

# Function to handle the conversion process
def convert_files(to_format):
    # Make sure script directory exists
    os.makedirs(script_dir, exist_ok=True)

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
