import torch
import torch.optim as optim
import torch.nn as nn
from Unet import UNet
from data_loader import *
import torch.nn.functional as F

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 超参数设置
num_epochs = 10
learning_rate = 0.001

# 新建一个Unet 对象并移动到GPU（如果有的话）
model = UNet(in_channels=1, out_channels=1).to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary segmentation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练数据的文件列表
train_list = load_data_list('data_set/Synapse/lists/lists_Synapse/train.txt')
# 测试数据的文件列表
test_list = load_data_list('data_set/Synapse/lists/lists_Synapse/test_vol.txt')

# 加载训练和测试数据集
train_dataset = SynapseDataset(train_list, 'data_set/Synapse/train_npz')
test_dataset = SynapseDataset(test_list, 'data_set/Synapse/test_vol_h5')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

for images, labels in train_loader:
    print(images.shape,labels.shape)

    
for images, labels in test_loader:
    print(images.shape,labels.shape)




# 开始训练
print('开始训练')
for epoch in range(num_epochs):
    model.train()
    
    running_loss = 0.0
    for images, labels in train_loader:
        # print(images.shape,labels.shape)
        # 将数据移动到GPU（如果有的话）
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)

        # 将标签转换为浮点类型
        labels = labels.float()

        # 调整标签尺寸以匹配输出
        labels_resized = F.interpolate(labels.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False)

        # 调整标签尺寸以匹配输出的形状 [8, 1, 256, 256]
        labels_resized = labels_resized.squeeze(1).unsqueeze(1)
        # outputs = torch.sigmoid(outputs)
        labels_resized = torch.sigmoid(outputs)
        loss = criterion(outputs, labels_resized.float())
        print(loss.item())

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')


    # Validation step (optional)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            # 将数据移动到GPU（如果有的话）
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)

            # 将标签转换为浮点类型
            labels = labels.float()

            # 调整标签尺寸以匹配输出
            labels_resized = F.interpolate(labels.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False)

            # 调整标签尺寸以匹配输出的形状 [8, 1, 256, 256]
            labels_resized = labels_resized.squeeze(1).unsqueeze(1)

            loss = criterion(outputs, labels_resized.float())
            val_loss += loss.item()
    print(f'Validation Loss: {val_loss/len(test_loader)}')

def dice_coeff(pred, target):
    smooth = 1.0
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    
    return dice.mean()

# Evaluation loop
model.eval()
dice_score = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        # 将数据移动到GPU（如果有的话）
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        dice_score += dice_coeff(outputs, labels).item()
print(f'Dice Score: {dice_score/len(test_loader)}')
