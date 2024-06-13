import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class SynapseDataset(Dataset):
    def __init__(self, data_list, data_dir, transform=None):
        self.data_list = data_list
        self.data_dir = data_dir
        self.transform = transform
        # self.resize = transforms.Resize((512, 512))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_name = self.data_list[idx]+''
        file_path = os.path.join(self.data_dir, file_name)
        # print(file_path)
        if file_path.endswith('.npz'):
            data = np.load(file_path)
            image = data['image']
            label = data['label']
        elif file_path.endswith('.h5'):
            with h5py.File(file_path, 'r') as f:
                image = np.array(f['image'])
                label = np.array(f['label'])
        else:
            raise ValueError("Unsupported file format")
        
        if self.transform:
            image, label = self.transform(image, label)
        
        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label

def load_data_list(file_path):
    with open(file_path, 'r') as f:
        data_list = f.read().splitlines()
    return data_list

# Example usage
# train_list = load_data_list('data_set/Synapse/lists/lists_Synapse/train.txt')
# test_list = load_data_list('data_set/Synapse/lists/lists_Synapse/test_vol.txt')

# train_dataset = SynapseDataset(train_list, 'data_set/Synapse/train_npz')
# test_dataset = SynapseDataset(test_list, 'data_set/Synapse/test_vol_h5')

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
