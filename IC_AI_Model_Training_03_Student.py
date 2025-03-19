import torch
import os


def train_yolov5(
        data_yaml: str,
        weights: str = 'yolov5s.pt',
        img_size: int = 640,
        batch_size: int = 16,
        epochs: int = 50,
        device: str = None,
        project: str = 'runs/train',
        name: str = 'exp',
        exist_ok: bool = False
):
    """
    Train a custom YOLOv5 model.

    Args:
        data_yaml (str): Path to the dataset YAML file.
        weights (str): Path to pretrained weights (default: yolov5s.pt).
        img_size (int): Image size (default: 640).
        batch_size (int): Batch size (default: 16).
        epochs (int): Number of training epochs (default: 50).
        device (str): Device to use ('cpu', 'cuda', or None to auto-detect).
        project (str): Path to save training results.
        name (str): Name of the training run.
        exist_ok (bool): Whether to overwrite existing results (default: False).
    """
    # Set device (auto-detect if None)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build the training command
    command = f"""
    export GIT_PYTHON_REFRESH=quiet &&
    cd /home/wheeltec/PycharmProjects/PythonProject/Yolo/IC_AI_Model_Training &&
    python /home/wheeltec/PycharmProjects/PythonProject/Yolo/IC_AI_Model_Training/train.py \
        --img {img_size} \
        --batch {batch_size} \
        --epochs {epochs} \
        --data {data_yaml} \
        --weights {weights} \
        --device {device} \
        --project {project} \
        --name {name} \
        {'--exist-ok' if exist_ok else ''}
    """

    # Run the training command
    print(f"Running command: {command}")
    os.system(command)


# Example usage
if __name__ == '__main__':
    # Path to dataset YAML file
    data_yaml_path = '/home/wheeltec/PycharmProjects/PythonProject/Yolo/IC_AI_Model_Training_03_Student.yaml'  # Replace with your dataset YAML file path

    # Path to pretrained weights (e.g., yolov5s.pt for COCO-trained weights)
    pretrained_weights = 'yolov5s.pt'  # Use yolov5n.pt, yolov5m.pt, etc., if needed

    # Call the train function
    train_yolov5(
        data_yaml=data_yaml_path,
        weights=pretrained_weights,
        img_size=640,  # Image size
        batch_size=16,  # Batch size
        epochs=10,  # Number of epochs
        device=None,  # Auto-detect device (GPU or CPU)
        project='/home/wheeltec/PycharmProjects/PythonProject/Yolo/IC_AI_Model_Export',  # Save training results in this folder
        name='IC_AI_model_03_Student',  # Name of the training experiment
        exist_ok=True  # Overwrite existing folder if exists
    )