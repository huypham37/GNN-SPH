import nbformat
import sys

def validate_notebook(notebook_path):
    try:
        # Try to read and parse the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            print(f"✓ Notebook '{notebook_path}' is valid!")
            return True
    except Exception as e:
        print(f"✗ Error in notebook '{notebook_path}':")
        print(f"  {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_notebook.py path/to/notebook.ipynb")
    else:
        validate_notebook(sys.argv[1]) 