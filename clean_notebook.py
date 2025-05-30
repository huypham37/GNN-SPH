import nbformat
import json
import sys

def clean_and_save_notebook(notebook_path):
    try:
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Create a clean filename
        clean_path = notebook_path.replace('.ipynb', '_clean.ipynb')
        
        # Save with proper formatting
        with open(clean_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
            
        print(f"✓ Created cleaned notebook: {clean_path}")
        return True
    except Exception as e:
        print(f"✗ Error cleaning notebook:")
        print(f"  {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_notebook.py path/to/notebook.ipynb")
    else:
        clean_and_save_notebook(sys.argv[1]) 