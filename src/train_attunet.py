import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.att_unet import *
from data_set import *
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

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
model = AttU_Net(1,1)
model_name = 'AttU_Net'
# 加载模型
# weights = torch.load('./pth/best_model.pth')
# model.load_state_dict(weights)

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

# 训练数据可视化
def visualization(num_epochs,train_losses,val_losses,train_dices,val_dices,train_ious,val_ious):
    # 绘制损失和指标变化曲线
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b', label='Train Loss')
    plt.plot(epochs, val_losses, 'r', label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_dices, 'b', label='Train Dice')
    plt.plot(epochs, val_dices, 'r', label='Val Dice')
    plt.title('Dice Coefficient')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_ious, 'b', label='Train IoU')
    plt.plot(epochs, val_ious, 'r', label='Val IoU')
    plt.title('IoU')
    plt.legend()

    plt.savefig(f'./train_result_visualization/{model_name}/{model_name}_training_metrics_epochs_{num_epochs}.png')
    
def train_log(epoch_num,train_loss,train_dice,train_iou,val_loss,val_dice,val_iou,log_file_name):
    with open(f'./logs/{log_file_name}.log', 'a') as log_file:
        log_file.write(f"train epoch : {epoch_num+1}\n")

        log_file.write(f"train_loss: {train_loss}\n")
        log_file.write(f"train_dice: {train_dice}\n")
        log_file.write(f"train_iou: {train_iou}\n")

        log_file.write(f"val_loss: {val_loss}\n")
        log_file.write(f"val_dice: {val_dice}\n")
        log_file.write(f"val_iou: {val_iou}\n")

        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'{formatted_time}\n\n')


# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 预处理标签
        labels = torch.clamp(labels, 0, 1)  # 将标签值限制在[0, 1]范围内

        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dice += calculate_dice(outputs, labels)
        running_iou += calculate_iou(outputs, labels)
    return running_loss / len(dataloader), running_dice / len(dataloader), running_iou / len(dataloader)

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

# 训练过程
num_epochs = 30
best_loss = float('inf')
best_val_dice = 0 

train_losses = []
val_losses = []
train_dices = []
val_dices = []
train_ious = []
val_ious = []

# test

# train_losses.append(0.1)
# val_losses.append(0.1)
# train_dices.append(0.1)
# val_dices.append(0.1)
# train_ious.append(0.1)
# val_ious.append(0.1)
# visualization(1,train_losses,val_losses,train_dices,val_dices,train_ious,val_ious)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss, train_dice, train_iou = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_dice, val_iou = validate_epoch(model, test_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Train IoU: {train_iou:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}, Validation IoU: {val_iou:.4f}")

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_dices.append(train_dice)
    val_dices.append(val_dice)
    train_ious.append(train_iou)
    val_ious.append(val_iou)

    visualization(epoch+1,train_losses,val_losses,train_dices,val_dices,train_ious,val_ious)

    # log 
    train_log(epoch,train_loss, train_dice, train_iou, val_loss, val_dice, val_iou,f'{model_name}_train_6_14')
    
    # 保存最好的模型
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), f'./pth/{model_name}_best_model_epoch_{epoch+1}.pth')
        print("Model saved!")
    elif best_val_dice < val_dice :
        best_val_dice = val_dice
        torch.save(model.state_dict(), f'./pth/{model_name}/{model_name}_best_val_dice__model_epoch_{epoch+1}.pth')
        print("Model saved!")


print("Training complete!")


