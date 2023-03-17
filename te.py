from astropy.utils.compat.futures._base import LOGGER
import torch

device = torch.device('cuda:3')  # 显卡序号信息
GPU_device = torch.cuda.get_device_properties(device)  # 显卡信息
print(f"YOLOv5 torch {torch.__version__} device {device} ({GPU_device.name}, {GPU_device.total_memory / 1024 ** 2}MB)\n")  # bytes to MB

# 也可以直接print()输出
