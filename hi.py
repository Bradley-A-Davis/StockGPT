import torch
print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)

import xformers
print("xFormers Version:", xformers.__version__)

