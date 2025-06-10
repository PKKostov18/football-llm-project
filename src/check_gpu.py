# src/check_gpu.py
import torch

print(f"PyTorch версия: {torch.__version__}")
print("-" * 30)

is_available = torch.cuda.is_available()
print(f"CUDA достъпно ли е: {is_available}")

if is_available:
    print(f"Брой GPU-та: {torch.cuda.device_count()}")
    print(f"Име на GPU 0: {torch.cuda.get_device_name(0)}")
else:
    print("\nPyTorch не намира CUDA.")
    print("Вероятно е инсталирана стандартната CPU-версия на PyTorch.")
    print("Трябва да я преинсталирате с версията, която поддържа CUDA.")