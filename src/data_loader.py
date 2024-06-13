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
            # print("image_size:{} , label_size:{}".format(image.shape,label.shape))

        elif file_path.endswith('.h5'):
            with h5py.File(file_path, 'r') as f:
                images = np.array(f['image'])
                labels = np.array(f['label'])
                for img in images:
                    image = img

                for lab in labels:
                    label = lab
                # print("image_size:{} , label_size:{}".format(image.shape,label.shape))
        else:
            raise ValueError("Unsupported file format")
        
        if self.transform:
            image, label = self.transform(image, label)
        
        # Convert to torch tensors
        # if file_path.endswith('.h5'):
        #     label = torch.tensor(label, dtype=torch.long)
        # else:
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        
        return image, label



class SynapseDataset_test(Dataset):
    def __init__(self, data_list, data_dir, transform=None):
        self.data_list = data_list
        self.data_dir = data_dir
        self.transform = transform
        # self.resize = transforms.Resize((512, 512))
        self.data_list_len = len(self.data_list)

        self.images =[]
        self.labels =[]
        

        for idx in range(self.data_list_len):
            file_name = self.data_list[idx]+''
            file_path = os.path.join(self.data_dir, file_name)
            # print(file_path)

            with h5py.File(file_path, 'r') as f:
                images = np.array(f['image'])
                labels = np.array(f['label'])
                for img in images:
                    self.images.append(img)

                for lab in labels:
                    self.labels.append(lab)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0) 
        # print("image_size:{} , label_size:{}".format(image.shape,label.shape))
        
        return image, label

def load_data_list(file_path):
    with open(file_path, 'r') as f:
        data_list = f.read().splitlines()
    return data_list

# Example usage



# train_list = load_data_list('data_set/Synapse/lists/lists_Synapse/train.txt')
# test_list = load_data_list('data_set/Synapse/lists/lists_Synapse/test_vol.txt')

# train_dataset = SynapseDataset(train_list, 'data_set/Synapse/train_npz')
# test_dataset = SynapseDataset_test(test_list, 'data_set/Synapse/test_vol_h5')

# print(test_dataset)

# train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# for images,labels in test_loader:
#     # print("test_loader",images.shape,labels.shape)
#     print("---")

# # for images,labels in train_loader:
# #     # print("train_loader",images.shape,labels.shape)
# #     print("---")