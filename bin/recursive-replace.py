import os
import subprocess
import argparse

def replace_in_files(root_dir, old_string, new_string, dry_run=False):
    """
    Recursively replaces occurrences of old_string with new_string in all files
    within the given directory tree.

    Args:
        root_dir (str): The root directory of the tree to traverse.
        old_string (str): The string to be replaced.
        new_string (str): The string to replace with.
        dry_run (bool): If True, only print the changes that would be made,
                        without actually modifying any files.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                if dry_run:
                    # Dry run: Print the sed command that would be executed
                    print(f"Dry run: sed -i 's/{old_string}/{new_string}/g' {filepath}")
                else:
                    # Execute the sed command
                    result = subprocess.run(
                        ["sed", "-i", f"s/{old_string}/{new_string}/g", filepath],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    if result.stderr:
                        print(f"Warning: sed reported an error on {filepath}: {result.stderr}")

            except subprocess.CalledProcessError as e:
                print(f"Error processing {filepath}: {e.stderr}")
            except Exception as e:
                print(f"An unexpected error occurred with {filepath}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replace text in files recursively.")
    parser.add_argument("root_dir", nargs="?", default=".", help="Root directory to search (default: .)")

