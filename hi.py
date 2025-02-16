import torch
import xformers
from xformers.ops import memory_efficient_attention

print("CUDA Available:", torch.cuda.is_available())  # Should return True
print("CUDA Version:", torch.version.cuda)  # Should be 12.1
print("PyTorch Version:", torch.__version__)  # Should be 2.1.0+cu121
print("xFormers Version:", xformers.__version__)  # Should print a version
