import torch
print(torch.cuda.is_available())      # should print True
print(torch.cuda.device_count())      # number of GPUs
print(torch.cuda.get_device_name(0))  # e.g. “GeForce RTX 3080”
