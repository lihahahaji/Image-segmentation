import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.u_net import UNet
from data_set import *

# 加载数据集文件列表
train_list = load_data_list('data_set/Synapse/lists/lists_Synapse/train.txt')
test_list = load_data_list('data_set/Synapse/lists/lists_Synapse/test_vol.txt')

# 创建数据集对象
train_dataset = SynapseDataset(train_list, 'data_set/Synapse/train_npz')
test_dataset = SynapseDataset_test(test_list, 'data_set/Synapse/test_vol_h5')

# 数据集的长度
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 定义模型
model = UNet(1, 1)

# 加载模型
weights = torch.load('best_model.pth')
model.load_state_dict(weights)

# 使用GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失，带logits
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 计算Dice系数的函数
def calculate_dice(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2. * intersection) / (union + 1e-6)  # 避免除以零
    return dice.mean().item()

# 计算IoU的函数
def calculate_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    iou = intersection / (union + 1e-6)  # 避免除以零
    return iou.mean().item()



# 验证函数
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # 预处理标签
            labels = torch.clamp(labels, 0, 1)  # 将标签值限制在[0, 1]范围内
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_dice += calculate_dice(outputs, labels)
            running_iou += calculate_iou(outputs, labels)
    return running_loss / len(dataloader), running_dice / len(dataloader), running_iou / len(dataloader)



# 验证模型
val_loss, val_dice, val_iou = validate_epoch(model, test_loader, criterion, device)
print(f"Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}, Validation IoU: {val_iou:.4f}")