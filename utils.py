import os
from datetime import datetime

def create_new_modelpath(save_path):
    """
    Create a unique directory for saving the model and results.

    Args:
        save_path (str): Base directory path.

    Returns:
        str: Newly created unique directory path.
    """
    timestamp = datetime.now().strftime('%m_%d_%H_%M')
    new_save_path = os.path.join(save_path, timestamp)
    os.makedirs(new_save_path, exist_ok=True)
    print("New path:", new_save_path)
    return new_save_path
