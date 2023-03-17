import torch.nn as nn
import torch.nn.functional as F
# from aug_model.Nonlocal import DotProduct


class LeNet(nn.Module):
    def __init__(self,num_classes=2):
        super(LeNet, self).__init__()
        # self.conv1 = nn.Conv3d(1, 32, 3,padding='same')
        self.conv1 = nn.Conv3d(1, 32, 3,padding=1)
        self.pool1 = nn.MaxPool3d(2,stride=2)
        # self.conv2 = nn.Conv3d(32, 64, 3,padding='same')
        self.conv2 = nn.Conv3d(32, 64, 3,padding=1)
        self.pool2 = nn.MaxPool3d(2,stride=2)
        # self.conv3 = nn.Conv3d(64, 64, 3,padding='same')
        self.conv3 = nn.Conv3d(64, 64, 3,padding=1)
        self.pool3 = nn.MaxPool3d(2,stride=2)
        # self.fc1 = nn.Linear(64*2*2*2, 1024)
        self.fc1 = nn.Linear(64 * 2 * 2 * 2, 500)
        self.dropout1=nn.Dropout(0.5)
        # self.fc2 = nn.Linear(1024, 32)
        self.fc2 = nn.Linear(500, 50)
        self.dropout2 = nn.Dropout(0.5)
        # self.fc3 = nn.Linear(32, num_classes)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))  # input(1, 16,16,16) output(16, 16,16,16)
        # print(x.shape)
        x = self.pool1(x)  # output(16, 8,8,8)
        # print(x.shape)

        # x=DotProduct(x)

        x = F.relu(self.conv2(x))  # output(32, 8,8,8)
        # print(x.shape)
        x = self.pool2(x)  # output(32, 4,4,4)
        # print(x.shape)

        # x = DotProduct(x)

        x = F.relu(self.conv3(x))  # output(64, 4,4,4)
        # print(x.shape)
        x = self.pool3(x)  # output(64, 2,2,2)
        # print(x.shape)

        # x = DotProduct(x)

        x = x.view(-1, 64*2*2*2)  # output(64*2*2*2)
        # print(x.shape)
        x = F.relu(self.fc1(x))  # output(1024)
        x = self.dropout1(x)
        # print(x.shape)
        x = F.relu(self.fc2(x))  # output(32)
        x = self.dropout2(x)
        # print(x.shape)
        x = self.fc3(x)  # output(2)
        return x
