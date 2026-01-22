import os
import torch
import timm
import glob
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class ColoredImagesDataset(Dataset):
    def __init__(self, data_dir, data_subdir, category=None):
        if category and category.lower() != "all":
            search_path = os.path.join(data_dir, category, data_subdir, "*.jpg")
        else:
            search_path = os.path.join(data_dir, "**", data_subdir, "*.jpg")

        self.image_paths = glob.glob(search_path, recursive=True)
        print(self.image_paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        print(f"Found {len(self.image_paths)} images matching {search_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), img_path


def extract_and_save(
    model_name, data_dir, data_subdir, save_path, category, device, batch_size=256
):
    dataset = ColoredImagesDataset(data_dir, data_subdir, category=category)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {data_dir} for category {category}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Loading model {model_name}...")
    model = (
        timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    )

    target_layer = model.blocks[-3]  # TODO: which layer?

    activations, metadata = [], []

    def hook_fn(m, i, o):
        if len(o.shape) == 3:  # [batch, tokens, hidden]
            # TODO: all tokens or one?
            # o = o.reshape(-1, o.shape[-1]) # include all tokens
            o = o[:, 0, :]  # include only the first token
        else:
            raise ValueError("Expected 3 dims.")
        activations.append(o.detach().cpu())

    handle = target_layer.register_forward_hook(hook_fn)

    print("Extracting activations...")
    with torch.no_grad():
        for imgs, paths in tqdm(loader, desc=f"Extracting {model_name}"):
            model(imgs.to(device))
            metadata.extend(paths)
    handle.remove()

    result = {
        "activations": torch.cat(activations, dim=0),
        "metadata": metadata,
        "model_name": model_name,
        "category": category,
    }
    torch.save(result, save_path)
    print(f"Saved {result['activations'].shape[0]} vectors to {save_path}")
    return result


def prepare_data(data_dir, data_subdir, category, model_type, save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_type == "clip":
        model_name = "vit_base_patch16_clip_224.openai"
    elif model_type == "dino":
        model_name = "vit_small_patch16_224.dino"
    else:
        model_name = model_type  # Allow passing direct timm model name

    if save_path is None:
        cat_str = category if category else "all"
        save_path = f"activations_{model_type}_{cat_str}.pt"

    if os.path.exists(save_path):
        print(f"Loading CACHED activations from: {save_path}")
        try:
            data = torch.load(save_path, map_location="cpu")
            return data["activations"]
        except Exception as e:
            print(f"Cache corrupted ({e}), re-extracting...")

    data = extract_and_save(
        model_name, data_dir, data_subdir, save_path, category=category, device=device
    )
    return data["activations"]
