import geopandas as gpd
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

class TextureDataset(Dataset):
    def __init__(self, image_dir, gdf_path, id_list=None, index_col='REFGDONSG', class_col='Type_TETIS_23', mean=None, std=None):
        
        self.index_col=index_col
        self.class_col=class_col
        self.gdf = gpd.read_file(gdf_path).set_index(self.index_col)
        self.image_dir = image_dir

        all_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(".jpg")
        ])

        if id_list is not None:
            id_set = set(id_list)
            self.image_files = [
                f for f in all_files
                if int(os.path.splitext(f)[0]) in id_set
            ]
        else:
            self.image_files = all_files

        base = [transforms.ToTensor()]
        if mean is not None and std is not None:
            base.append(transforms.Normalize(mean=[mean], std=[std]))
        self.transform = transforms.Compose(base)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        obj_id = int(os.path.splitext(fname)[0])

        obj_label = int(self.gdf.loc[obj_id][self.class_col])

        img_path = os.path.join(self.image_dir, fname)
        img = Image.open(img_path).convert("L")  # "L" = grayscale, single channel
        
        img = np.array(img, dtype=np.float32)

        #img = largest_rectangle(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = Image.fromarray(img)

        img = self.transform(img)
        return img, obj_label, obj_id
    
class TextureEncoder(nn.Module):
    def __init__(self, out_dim=128, padding=1, kernel_size=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),

            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),

            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
        )

        self.projection = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.flatten(1)
        return self.projection(x)
    
class TextureDecoder(nn.Module):
    def __init__(self, in_dim=128, kernel_size=3, padding=1):
        super().__init__()
        self.projection = nn.Linear(in_dim, 128)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),

            nn.ConvTranspose2d(64, 32, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),

            nn.ConvTranspose2d(32, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid()   # output in [0, 1] to match normalized input
        )

    def forward(self, z, target_size):
        x = self.projection(z)                          
        x = x.unsqueeze(-1).unsqueeze(-1)               
        x = F.interpolate(x, size=target_size,
                          mode='bilinear',
                          align_corners=False)          
        return self.features(x)                         
    
class TextureAutoencoder(nn.Module):
    def __init__(self, out_dim=128, kernel_size=3, padding=1):
        super().__init__()
        self.encoder = TextureEncoder(out_dim=out_dim,
                                      kernel_size=kernel_size,
                                      padding=padding)
        self.decoder = TextureDecoder(in_dim=out_dim,
                                      kernel_size=kernel_size,
                                      padding=padding)

    def forward(self, x):
        target_size = (x.shape[-2], x.shape[-1])   
        z = self.encoder(x)                        
        x_hat = self.decoder(z, target_size)       
        return x_hat, z
    
def train_autoencoder(
        image_dir,
        gdf_path,
        train_ids,
        train_mean,
        train_std,
        out_dim=128,
        num_epochs=30,
        lr=1e-3,
        save_path="autoencoder.pt"
    ):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = TextureDataset(image_dir, gdf_path,
                                   id_list=train_ids,
                                   mean=train_mean, std=train_std)

    train_loader = DataLoader(train_dataset, batch_size=8,
                              shuffle=True, collate_fn=collate_variable)
    
    model     = TextureAutoencoder(out_dim=out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        n_train    = 0
        for imgs, _, _ in train_loader:
            for img in imgs:
                img = img.unsqueeze(0).to(device)       # (1, 1, H, W)
                optimizer.zero_grad()
                img_hat, _ = model(img)
                loss = criterion(img_hat, img)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_train    += 1

    
    torch.save(model.state_dict(), save_path)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.5f}")
    return model

def extract_features(
        image_dir,
        gdf_path,
        id_list,
        train_mean,
        train_std,
        model_path="autoencoder.pt",
        out_dim=128
    ):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the full autoencoder, then keep only the encoder
    model = TextureAutoencoder(out_dim=out_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    encoder = model.encoder.to(device)
    encoder.eval()

    dataset = TextureDataset(image_dir, gdf_path,
                             id_list=id_list,
                             mean=train_mean, std=train_std)
    loader  = DataLoader(dataset, batch_size=8,
                         shuffle=False, collate_fn=collate_variable)

    rows = []
    with torch.no_grad():
        for imgs, labels, obj_ids in loader:
            for img, label, obj_id in zip(imgs, labels, obj_ids):
                img  = img.unsqueeze(0).to(device)
                feat = encoder(img).squeeze(0).cpu().numpy()
                rows.append({
                    "id":    obj_id,
                    "label": label,
                    **{f"f{i}": feat[i] for i in range(len(feat))}
                })

    df = (pd.DataFrame(rows)
            .sort_values("id")
            .reset_index(drop=True))
    return df

def get_norm_vals(data_paths, metrics='mean_std'):
    data = []
    for path in data_paths:
        data.append(np.load(path))
    
    concat_data = np.concatenate(data)

    norm_vals_1 = []
    norm_vals_2 = []

    for band in range(concat_data.shape[1]):
        if metrics=='min_max':
            norm_vals_1.append(np.min(concat_data[:,band,:]))
            norm_vals_2.append(np.max(concat_data[:,band,:]))
        elif metrics=='mean_std':
            norm_vals_1.append(np.mean(concat_data[:,band,:]))
            norm_vals_2.append(np.std(concat_data[:,band,:]))

    return norm_vals_1, norm_vals_2

def normalize(data, norm_vals_1, norm_vals_2, norm_method='mean_std'):
    for band in range(data.shape[1]):
        if norm_method == 'mean_std':
            data[:,band,:] = (data[:,band,:] - norm_vals_1[band]) / norm_vals_2[band]
        elif norm_method == 'min_max':
            data[:,band,:] = (data[:,band,:] - norm_vals_1[band]) / (norm_vals_2[band]- norm_vals_1[band])
    return data

def collate_variable(batch):
    imgs    = [item[0] for item in batch]
    labels  = [item[1] for item in batch]
    obj_ids = [item[2] for item in batch]
    return imgs, labels, obj_ids

def compute_mean_std(dataset):
    """Compute per-channel mean and std over the training set."""
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_variable)
    mean = 0.0
    std  = 0.0
    n_pixels = 0
    for imgs, _, _ in loader:
        for img in imgs:
            mean     += img.mean().item() * img.numel()
            n_pixels += img.numel()
    mean /= n_pixels
    for imgs, _, _ in loader:
        for img in imgs:
            std += ((img - mean) ** 2).sum().item()
    std = (std / n_pixels) ** 0.5
    return mean, std