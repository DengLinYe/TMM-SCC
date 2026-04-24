import torch
import transformers

print(f"PyTorch: {torch.__version__} (CUDA: {torch.version.cuda})")
print(f"Transformers: {transformers.__version__}")
print(f"Device: {torch.cuda.get_device_name(0)}")
