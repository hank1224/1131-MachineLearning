import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 單輸入單輸出

    def forward(self, x):
        return self.linear(x)

class EWMA:
    def __init__(self, lambda_value):
        self.lambda_value = lambda_value

    def update(self, current_value, previous_value):
        return self.lambda_value * current_value + (1 - self.lambda_value) * previous_value

class Measurement:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

class ProcessControl:
    def __init__(self, model, ewma, target_height):
        self.model = model
        self.ewma = ewma
        self.target_height = target_height
        self.previous_state = 0

    def pre_process(self, pre_thk):
        print(f"Pre-process thickness: {pre_thk}")
        return pre_thk

    def process(self, pre_thk):
        dep_time = (self.target_height - pre_thk) / 2  # 假設Dep_Rate = 2
        print(f"Processing with deposition time: {dep_time}")
        return dep_time

    def post_process(self, measured_thk):
        print(f"Post-process measured thickness: {measured_thk}")
        updated_thk = self.ewma.update(measured_thk, self.previous_state)
        self.previous_state = updated_thk
        return updated_thk

    def run(self, pre_thk):
        pre_thk = self.pre_process(pre_thk)
        dep_time = self.process(pre_thk)
        measured_thk = pre_thk + 2 * dep_time  # 假設Dep_Rate = 2
        updated_thk = self.post_process(measured_thk)
        print(f"Updated thickness after EWMA: {updated_thk}")

# 生成隨機訓練數據
np.random.seed(0)
x_train = np.random.rand(100, 1).astype(np.float32) * 10  # 隨機生成0到10之間的數字
y_train = 1.91 * x_train + np.random.randn(100, 1).astype(np.float32)  # y = 1.91x + 隨機噪聲

# 將數據轉換為PyTorch張量
x_train_tensor = torch.from_numpy(x_train)
y_train_tensor = torch.from_numpy(y_train)

# 初始化模型和優化器
model = LinearRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 訓練模型
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')

# 初始化控制系統
lambda_value = 0.3
target_height = 20
ewma = EWMA(lambda_value)
control_system = ProcessControl(model, ewma, target_height)

# 執行製程控制
initial_pre_thk = 6
control_system.run(initial_pre_thk)