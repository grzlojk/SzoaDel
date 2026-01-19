import os
import torch
import timm
import fiftyone as fo
import fiftyone.zoo as foz
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class FiftyOneTorchDataset(Dataset):
    def __init__(self, fo_dataset):
        self.filepaths = fo_dataset.values("filepath")
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        img = Image.open(filepath).convert('RGB')
        return self.transform(img)

def prepare_data(num_samples, save_path="lizard_activations.pt", batch_size=32):
    """
    Sprawdza, czy aktywacje istnieją. Jeśli nie, pobiera dataset, 
    uruchamia ViT i zapisuje tensory.
    """
    
    # 1. Sprawdzenie cache
    if os.path.exists(save_path):
        print(f"Loading activations from existing file: {save_path}")
        return torch.load(save_path)

    print(f"Activations not found. Downloading dataset with {num_samples} samples...")
    
    # 2. FiftyOne Download
    dataset_name = f"open-images-lizards-{num_samples}"
    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
    else:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="validation",
            label_types=["detections"],
            classes=["Lizard"],
            max_samples=num_samples,
            shuffle=True,
            dataset_name=dataset_name
        )
    
    # 3. Model ViT & Hook
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading ViT model...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    model.eval().to(device)
    
    # Hook na konkretną warstwę (zgodnie z oryginałem)
    target_layer = model.blocks[-3].mlp.fc1
    extracted_activations = []

    def hook_fn(module, input, output):
        extracted_activations.append(output.detach().cpu())

    handle = target_layer.register_forward_hook(hook_fn)
    
    # 4. Ekstrakcja
    torch_dataset = FiftyOneTorchDataset(dataset)
    dataloader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=2)
    
    print("Extracting activations...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            _ = model(batch)
            
    handle.remove()
    
    # 5. Przetwarzanie i zapis
    # [Total_Images, 197, 768] -> [N * 197, 768]
    all_acts = torch.cat(extracted_activations, dim=0)
    flattened_acts = all_acts.reshape(-1, all_acts.shape[-1])
    
    print(f"Saving {flattened_acts.shape[0]} vectors to {save_path}")
    torch.save(flattened_acts, save_path)
    
    return flattened_acts