import torch.optim as optim
import torch.nn as nn
# from Unet import UNet
from net_work import *
from data_loader import *
import torch.nn.functional as F
from tqdm import tqdm 
from datetime import datetime

# use gpu
cuda_available = True

# 超参数设置
learning_rate = 0.01

# 训练数据的文件列表
train_list = load_data_list('data_set/Synapse/lists/lists_Synapse/train.txt')
# 测试数据的文件列表
test_list = load_data_list('data_set/Synapse/lists/lists_Synapse/test_vol.txt')

# 加载训练和测试数据集
train_dataset = SynapseDataset(train_list, 'data_set/Synapse/train_npz')
test_dataset = SynapseDataset_test(test_list, 'data_set/Synapse/test_vol_h5')

# 数据集的长度
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 创建模型
model = AttU_Net(1,1)


# 定义损失函数和优化器
loss_fn = nn.BCELoss()  # Binary Cross Entropy Loss for binary segmentation
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if cuda_available:
    model = model.cuda()
    loss_fn = loss_fn.cuda()

# 加载权重
# weights = torch.load('./pth/Unet.pth')
# model.load_state_dict(weights)

# 记录训练的次数
total_train_step = 0

# 记录测试的次数
total_test_step = 0

# 训练的轮数
epoch = 100

# Dice系数计算函数
def dice_coeff(pred, target):
    smooth = 1.0  # 防止分母为0
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

# IoU计算函数
def iou(pred, target):
    smooth = 1.0  # 防止分母为0
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# 开始训练
for i in range(epoch):
    print("--------第 {} 轮训练开始--------".format(i+1))
    model.train()
    total_train_step = 0 

    for data in tqdm(train_loader, desc="Training"): 
        images,labels = data
        if cuda_available:
            images = images.cuda()
            labels = labels.cuda()

        # labels_resized = labels.squeeze(1).unsqueeze(1)

        outputs = model(images)

        labels_resized = torch.sigmoid(labels)
        outputs = torch.sigmoid(outputs)
        loss = loss_fn(outputs,labels_resized)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # print("训练次数：{}, loss：{}".format(total_train_step , loss.item()))

        # if(total_train_step%100 ==0 ):
        #     print("训练次数：{}, loss：{}".format(total_train_step , loss.item()))

        # break

    # 开始测试模型
    model.eval()
    total_test_loss = 0
    total_dice = 0
    total_iou = 0
    # print('开始测试模型')

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            images,labels = data

            if(cuda_available):
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            # labels_resized = labels.squeeze(1).unsqueeze(1)

            labels_resized = torch.sigmoid(labels)
            outputs = torch.sigmoid(outputs)

            loss = loss_fn(outputs,labels_resized)
            total_test_loss += loss.item()

            # 计算Dice系数和IoU
            dice = dice_coeff(outputs, labels_resized)
            total_dice += dice.item()

            # iou_score = iou(outputs, labels_resized)
            # total_iou += iou_score.item()

    avg_test_loss = total_test_loss / len(test_loader)
    avg_dice = total_dice / len(test_loader)
    # avg_iou = total_iou / len(test_loader)

    print("整体测试集上的Loss:{}".format(avg_test_loss))
    print("整体测试集上的Dice系数:{}".format(avg_dice))
    # print("整体测试集上的IoU:{}".format(avg_iou))

    total_test_step += 1

    torch.save(model.state_dict(), "./pth/Att_Unet_epoch_{}.pth".format(i))
    print("模型已保存")

    # 记录训练日志
    with open('./logs/train.log', 'a') as log_file:
        # 写入训练轮次信息
        log_file.write(f'------------第 {i+1} 轮训练------------\n')
        log_file.write(f'整体测试集上的Loss: {avg_test_loss}\n')
        log_file.write(f'整体测试集上的Dice系数: {avg_dice}\n')
        log_file.write(f'模型权重已保存到: ./pth/Unet_epoch_{i}.pth\n')

        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'{formatted_time}\n\n')

        
