import torch

# 检查可用的GPU数量
gpu_count = torch.cuda.device_count()
print(f"Available GPU count: {gpu_count}")
print(torch.cuda.is_available())