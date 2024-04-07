import os
import torch


checkpoint_root = "sac/checkpoint/"

def load_checkpoints(checkpoint_root, in_dim=22664):
    data = {
        "param": [],   # List to store parameter tensors
        "traj": [],  # List to store trajectory data
        "task": [],  # List to store task numbers
    }
    pdata = []
    for filename in os.listdir(checkpoint_root):
        if filename.endswith(".pth"):
            filepath = os.path.join(checkpoint_root, filename)
            checkpoints = torch.load(filepath, map_location='cpu')
            for checkpoint in checkpoints:
                param = []
                for key in checkpoint.keys():
                    param.append(checkpoint[key].data.reshape(-1))
                param = torch.cat(param, 0)
                if len(param) == in_dim:
                    pdata.append(param)
    data["param"] = torch.stack(pdata)
    return data

data = load_checkpoints(checkpoint_root)
torch.save(data, "data.pt")