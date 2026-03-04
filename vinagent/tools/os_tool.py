import os
import shutil
import subprocess
import webbrowser
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
from vinagent.register import primary_function

# Safety constraints
DANGEROUS_COMMANDS = [
    "rm -rf /",
    "rm -rf *",
    ":(){ :|:& };:",
    "mkfs",
    "> /dev/",
    "dd if=",
    "chmod -R 777",
    "chown -R",
    "curl",
    "wget",
    "nc -e",
    "sh -i",
    "bash -i",
    "python -c",
    "perl -e",
    "ruby -e",
    "socat",
    "poweroff",
    "reboot",
    "shutdown",
    "init 0",
    "init 6",
    "/dev/sda",
    "/dev/nvme",
]


@primary_function
def create_file(file_path: str, content: str = "") -> str:
    """
    Creates a new file at the specified path with optional content.

    Args:
        file_path (str): The path where the file should be created.
        content (str, optional): The initial content to write to the file. Defaults to "".

    Returns:
        str: A success message if the file was created, or an error message otherwise.
    """
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Successfully created file: {file_path}"
    except Exception as e:
        return f"Error creating file {file_path}: {str(e)}"


@primary_function
def read_file(file_path: str) -> str:
    """
    Reads the content of the file at the specified path.

    Args:
        file_path (str): The path to the file to be read.

    Returns:
        str: The content of the file if successful, or an error message otherwise.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found at {file_path}"
        if not path.is_file():
            return f"Error: {file_path} is not a file"
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file {file_path}: {str(e)}"


@primary_function
def update_file_content(file_path: str, content: str, mode: str = "append") -> str:
    """
    Updates the content of a file by either appending to it or overwriting it.

    Args:
        file_path (str): The path to the file to be updated.
        content (str): The content to write to the file.
        mode (str, optional): The update mode. Must be 'append' or 'overwrite'. Defaults to "append".

    Returns:
        str: A success message if the file was updated, or an error message otherwise.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found at {file_path}"

        if mode == "append":
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
        elif mode == "overwrite":
            path.write_text(content, encoding="utf-8")
        else:
            return f"Error: Invalid mode '{mode}'. Use 'append' or 'overwrite'."

        return f"Successfully updated file: {file_path} (mode: {mode})"
    except Exception as e:
        return f"Error updating file {file_path}: {str(e)}"


@primary_function
def delete_file(file_path: str) -> str:
    """
    Deletes the file at the specified path.

    Args:
        file_path (str): The path to the file to be deleted.

    Returns:
        str: A success message if the file was deleted, or an error message otherwise.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found at {file_path}"
        if not path.is_file():
            return f"Error: {file_path} is not a file"
        path.unlink()
        return f"Successfully deleted file: {file_path}"
    except Exception as e:
        return f"Error deleting file {file_path}: {str(e)}"


@primary_function
def copy_file(source_path: str, destination_path: str) -> str:
    """
    Copies a file from the source path to the destination path.

    Args:
        source_path (str): The path to the source file.
        destination_path (str): The destination path (file or directory).

    Returns:
        str: A success message if the file was copied, or an error message otherwise.
    """
    try:
        shutil.copy2(source_path, destination_path)
        return f"Successfully copied {source_path} to {destination_path}"
    except Exception as e:
        return f"Error copying file from {source_path} to {destination_path}: {str(e)}"


@primary_function
def move_file(source_path: str, destination_path: str) -> str:
    """
    Moves or renames a file from the source path to the destination path.

    Args:
        source_path (str): The path to the source file.
        destination_path (str): The destination path (file or directory).

    Returns:
        str: A success message if the file was moved, or an error message otherwise.
    """
    try:
        shutil.move(source_path, destination_path)
        return f"Successfully moved {source_path} to {destination_path}"
    except Exception as e:
        return f"Error moving file from {source_path} to {destination_path}: {str(e)}"


@primary_function
def create_folder(folder_path: str) -> str:
    """
    Creates a new folder at the specified path, including any necessary parent folders.

    Args:
        folder_path (str): The path of the folder to be created.

    Returns:
        str: A success message if the folder was created, or an error message otherwise.
    """
    try:
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        return f"Successfully created folder: {folder_path}"
    except Exception as e:
        return f"Error creating folder {folder_path}: {str(e)}"


@primary_function
def delete_folder(folder_path: str) -> str:
    """
    Deletes a folder and all of its contents.

    Args:
        folder_path (str): The path of the folder to be deleted.

    Returns:
        str: A success message if the folder was deleted, or an error message otherwise.
    """
    try:
        path = Path(folder_path)
        if not path.exists():
            return f"Error: Folder not found at {folder_path}"
        if not path.is_dir():
            return f"Error: {folder_path} is not a folder"
        shutil.rmtree(path)
        return f"Successfully deleted folder: {folder_path}"
    except Exception as e:
        return f"Error deleting folder {folder_path}: {str(e)}"


@primary_function
def list_directory(directory_path: str = ".") -> Union[List[str], str]:
    """
    Lists the contents of a directory.

    Args:
        directory_path (str, optional): The path to the directory to list. Defaults to ".".

    Returns:
        Union[List[str], str]: A list of file and folder names if successful, or an error message otherwise.
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            return f"Error: Directory not found at {directory_path}"
        if not path.is_dir():
            return f"Error: {directory_path} is not a directory"
        return [item.name for item in path.iterdir()]
    except Exception as e:
        return f"Error listing directory {directory_path}: {str(e)}"


@primary_function
def open_browser(url: str) -> str:
    """
    Opens the default web browser to the specified URL.

    Args:
        url (str): The URL to open in the browser.

    Returns:
        str: A success message if the browser was opened, or an error message otherwise.
    """
    try:
        webbrowser.open(url)
        return f"Successfully opened browser to: {url}"
    except Exception as e:
        return f"Error opening browser to {url}: {str(e)}"


@primary_function
def execute_bash(command: str, timeout: int = 30) -> str:
    """
    Executes a shell command and returns the output. Safety checks are performed to block dangerous commands.
    Every terminal command is wrapped in a subprocess with a strict 30-second timeout.

    IMPORTANT: The agent is explicitly instructed to perform pre-verification of the command
    to reduce the risk of unintended side effects.

    Args:
        command (str): The shell command to execute.
        timeout (int, optional): The maximum time in seconds to wait for the command to complete. Defaults to 30.

    Returns:
        str: The standard output and standard error of the command, or an error message if the command was blocked or failed.
    """
    # Safety Check
    cmd_lower = command.lower()
    for dangerous in DANGEROUS_COMMANDS:
        if dangerous in cmd_lower:
            return f"Error: Command blocked for safety reasons. '{dangerous}' is forbidden."

    try:
        result = subprocess.run(
            command, shell=True, text=True, capture_output=True, timeout=timeout
        )
        output = result.stdout if result.stdout else ""
        error = result.stderr if result.stderr else ""

        if result.returncode == 0:
            return output if output else "Command executed successfully with no output."
        else:
            return f"Command failed with return code {result.returncode}.\nOutput: {output}\nError: {error}"

    except subprocess.TimeoutExpired:
        return f"Error: Command execution timed out after {timeout} seconds."
    except Exception as e:
        return f"Error executing command: {str(e)}"
