import torch.nn as nn


class SimpleCNN_MLP(nn.Module):
    def __init__(self, input_channels, input_dim, hidden_dim, output_dim):
        super(SimpleCNN_MLP, self).__init__()

        # 第一層 1D 卷積層 (kernel_size=7, padding=3)
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=7, padding=3)
        # 第二層 1D 卷積層 (kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)

        self.relu = nn.ReLU()

        # Fully Connected Layers
        self.fc1 = nn.Linear(32 * input_dim, hidden_dim)  # 調整為 32 * input_dim
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        # 確保輸出符合 非負 & 和等於 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1D CNN
        x = self.relu(self.conv1(x))  # 第一層 1D 卷積
        x = self.relu(self.conv2(x))  # 第二層 1D 卷積

        x = x.view(x.size(0), -1)  # 展平 (flatten)

        # Fully Connected Layers
        x = self.relu(self.fc1(x))  # 第一層全連接層
        x = self.relu(self.fc2(x))  # 第二層全連接層
        x = self.fc3(x)  # 輸出層

        # 非負條件 & 和為 1
        x = self.sigmoid(x)
        x = x / x.sum(dim=1, keepdim=True)

        return x

