# from datetime import datetime
# current_time = datetime.now()
# formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
# print(formatted_time)

# with open('./logs/train.log', 'a') as log_file:
#     log_file.write(f'{formatted_time}\n\n')


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data_set import *

from PIL import Image
import numpy as np



train_list = load_data_list('data_set/Synapse/lists/lists_Synapse/train.txt')
test_list = load_data_list('data_set/Synapse/lists/lists_Synapse/test_vol.txt')

train_dataset = SynapseDataset(train_list, 'data_set/Synapse/train_npz')
test_dataset = SynapseDataset_test(test_list, 'data_set/Synapse/test_vol_h5')

print(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

transform_to_pil = transforms.ToPILImage()

# test_image_idx = 0
# test_label_idx = 0
# for images,labels in test_loader:
#     # print("test_loader",images.shape,labels.shape)
#     # print("---")
#     for image in images:
#         image_file = transform_to_pil(image)
#         save_path = f'./test_images/test_image_{test_image_idx}.png'
#         test_image_idx+=1
#         image_file.save(save_path)

#     for label in labels:
#         label_file = transform_to_pil(label)
#         save_path = f'./test_labels/test_label_{test_label_idx}.png'
#         test_label_idx+=1
#         label_file.save(save_path)

# # for images,labels in train_loader:
# #     # print("train_loader",images.shape,labels.shape)
# #     print("---")

test_image_idx = 0
test_label_idx = 0

for images, labels in test_loader:
    # Loop through each image and its corresponding label
    for image, label in zip(images, labels):
        # Convert tensors to PIL images
        image_pil = transform_to_pil(image)
        label_pil = transform_to_pil(label)
        
        # Get the dimensions of the images
        width, height = image_pil.size
        
        # Create a new image with double the width to hold both the image and label side by side
        combined_image = Image.new('RGB', (width * 2, height))
        
        # Paste the image and the label into the combined image
        combined_image.paste(image_pil, (0, 0))
        combined_image.paste(label_pil, (width, 0))
        
        # Save the combined image
        save_path = f'./test_images_labels/test_image_label_{test_image_idx}.png'
        combined_image.save(save_path)
        
        # Increment the index
        test_image_idx += 1
