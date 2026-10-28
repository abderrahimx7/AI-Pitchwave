import platform
import sys
try:
    import torch
except ImportError:
    print('PyTorch not installed in this Python environment:', sys.executable)
    raise SystemExit(1)
print('Python:', platform.python_version())
print('Executable:', sys.executable)
print('PyTorch:', torch.__version__)
print('Wheel CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
    a = torch.randn(128, 128, device='cuda', requires_grad=True)
    loss = (a @ a.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    print('PASS: CUDA matrix multiplication and backward pass')
else:
    print('CPU usable, but GPU training not ready. Send this output and nvidia-smi for diagnosis.')
    raise SystemExit(2)
