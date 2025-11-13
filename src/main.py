import torch


def main():
    print("hello world")
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. You can use GPU acceleration.")
    else:
        print("CUDA is not available. Running on CPU.")
    print("PyTorch version:", torch.__version__)
