# Kaggle: save the CLIP ViT-L/14 (OpenAI) weights in half precision so the web app can load them from a local file
# (downloading them from Hugging Face on the laptop's connection was too slow).
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
import open_clip, torch
m, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
torch.save({k: v.half() for k, v in m.state_dict().items()}, "/kaggle/working/clip_l14_openai_fp16.pt")
print("saved")
