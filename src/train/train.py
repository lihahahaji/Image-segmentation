import torch.optim as optim
import torch.nn as nn
# from Unet import UNet
from net_work import *
from data_set import *
import torch.nn.functional as F
# use gpu
cuda_available = True


# 超参数设置
learning_rate = 0.001

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

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)

# 创建模型
model = U_Net(1,1)



# 定义损失函数和优化器
loss_fn = nn.BCELoss()  # Binary Cross Entropy Loss for binary segmentation
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if cuda_available:
    model = model.cuda()
    loss_fn = loss_fn.cuda()

# 记录训练的次数
total_train_step = 0

# 记录测试的次数
total_test_step = 0

# 训练的轮数
epoch = 10




# 开始训练
for i in range(epoch):
    print("--------第 {} 轮训练开始--------".format(i+1))
    model.train()
    total_train_step = 0 

    for data in train_loader:
        images,labels = data
        # print(images.shape,labels.shape)
        if cuda_available:
            images = images.cuda()
            labels = labels.cuda()

        labels = labels
        # labels_resized = F.interpolate(labels.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False)
        labels_resized = labels.squeeze(1).unsqueeze(1)

        outputs = model(images)

        labels_resized = torch.sigmoid(labels_resized)
        outputs = torch.sigmoid(outputs)
        loss = loss_fn(outputs,labels_resized.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1

        if(total_train_step%100 ==0 ):
            print("训练次数：{}, loss：{}".format(total_train_step , loss.item()))
        # break



    # 开始测试模型
    model.eval()
    total_test_loss = 0

    with torch.no_grad():
        for data in test_loader:
            images,labels = data

            if(cuda_available):
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            labels_resized = labels.squeeze(1).unsqueeze(1)

            labels_resized = torch.sigmoid(labels_resized)
            outputs = torch.sigmoid(outputs)

            loss = loss_fn(outputs,labels_resized.float())
            total_test_loss += loss.item()
            # print(loss.item())

    print("整体测试集上的Loss:{}".format(total_test_loss))

    total_test_step += 1

    torch.save(model, "./pth/Unet_{}.pth".format(i))
    print("模型已保存")







# # 开始训练
# # Training loop
# print('开始训练')
# for epoch in range(num_epochs):
#     model.train()
    
#     running_loss = 0.0
#     for images, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(images)
        
#         # print(print(outputs.shape))
#         # 将标签转换为浮点类型
#         labels = labels.float()

#         # 调整标签尺寸以匹配输出
#         labels_resized = F.interpolate(labels.unsqueeze(1), size=(256, 256), mode='bilinear', align_corners=False)

#         # 调整标签尺寸以匹配输出的形状 [8, 1, 256, 256]
#         labels_resized = labels_resized.squeeze(1).unsqueeze(1)
#         loss = criterion(outputs, labels_resized.float())
#         print(loss)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
    
#     print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

#     # Validation step (optional)
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             outputs = model(images)
#             loss = criterion(outputs, labels.float())
#             val_loss += loss.item()
#     print(f'Validation Loss: {val_loss/len(test_loader)}')

# def dice_coeff(pred, target):
#     smooth = 1.0
#     pred = pred.contiguous()
#     target = target.contiguous()
    
#     intersection = (pred * target).sum(dim=2).sum(dim=2)
#     dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    
#     return dice.mean()

# # Evaluation loop
# model.eval()
# dice_score = 0.0
# with torch.no_grad():
#     for images, labels in test_loader:
#         outputs = model(images)
#         dice_score += dice_coeff(outputs, labels).item()
# print(f'Dice Score: {dice_score/len(test_loader)}')
