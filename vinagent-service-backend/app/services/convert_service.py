import os
from typing import Literal
import tempfile
import subprocess
from app.core.setting import get_settings

config = get_settings()

texbin_path = config.textbin_path

if texbin_path and texbin_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = texbin_path + os.pathsep + os.environ["PATH"]

class MarkdownConversionError(Exception):
    pass

def convert_markdown_to_file(
    markdown_content: str,
    output_format: Literal["pdf", "docx"]
) -> str:
    """
    Converts markdown content to the specified format (pdf or docx) using Pandoc.
    Returns the path to the generated output file.
    Raises MarkdownConversionError if the conversion fails.
    """
    if not markdown_content:
        raise ValueError("Markdown content cannot be empty")
    
    # Create temporary input markdown file
    input_filepath = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+', suffix=".md", delete=False, encoding='utf-8') as input_file:
            input_file.write(markdown_content)
            input_filepath = input_file.name  # Get the path to the temporary file

        # Create temporary output file path
        output_suffix = ".pdf" if output_format == "pdf" else ".docx"
        output_filepath = None
        with tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False) as output_file:
            output_filepath = output_file.name  # Get the path for the output file

        pandoc_command = ["pandoc", input_filepath, "-o", output_filepath]
        # Execute the Pandoc command
        # capture_output=True to get stdout/stderr, text=True for string output
        process = subprocess.run(pandoc_command, capture_output=True, text=True, check=False)

        # Check Pandoc's exit code
        if process.returncode != 0:
            error_details = process.stderr.strip() if process.stderr else "Unknown Pandoc error"
            print(f"Pandoc stderr: {process.stderr}")
            raise MarkdownConversionError(f"Pandoc conversion failed: {error_details}")

        return output_filepath
    except Exception as e:
        if 'output_filepath' in locals() and output_filepath and os.path.exists(output_filepath):
            os.remove(output_filepath)
        if isinstance(e, MarkdownConversionError):
            raise e  # Re-raise our custom error
        else:
            raise MarkdownConversionError(f"An internal error occurred during conversion: {e}") from e
        
    finally:
        if input_filepath and os.path.exists(input_filepath):
            os.remove(input_filepath)
        pass