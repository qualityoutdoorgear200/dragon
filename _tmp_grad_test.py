from torch.amp import GradScaler
s1 = GradScaler()
print("default device:", s1._scale.device)
s2 = GradScaler("cuda")
print("cuda device specified")
