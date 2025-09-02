import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

from model import *

# 加载训练数据集
train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=0, drop_last=True)

# 加载测试数据集
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

# 定义模型、优化器
model = ResNet101(10)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 损失函数
criterion = F.cross_entropy

# 训练轮数
num_epochs = 10

# 使用设备（如果有GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()  # 训练模式
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 打印每个epoch的平均训练损失
    avg_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}', end=' ')

    # === 测试阶段 ===
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():  # 不需要梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    # 计算平均测试损失和准确率
    avg_test_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    print(f'Test Loss: {avg_test_loss:.4f}, Test Acc: {accuracy:.2f}%')
