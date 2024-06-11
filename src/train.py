import torch.optim as optim
import torch.nn as nn


# Hyperparameters
num_epochs = 10
learning_rate = 0.001

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for binary segmentation
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

    # Validation step (optional)
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels.float())
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
        outputs = model(images)
        dice_score += dice_coeff(outputs, labels).item()
print(f'Dice Score: {dice_score/len(test_loader)}')
