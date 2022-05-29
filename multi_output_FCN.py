import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(128, 384, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.conv3 = nn.Conv2d(384, 768, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.conv4 = nn.Conv2d(768, 2048, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)


        self.fc1 = nn.Linear(679936, 39)#labels
        self.fc2 = nn.Linear(679936, 78)#labels
        self.fc3 = nn.Linear(679936, 1)#labels



    def forward(self, x):
        x = self.dropout(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(self.pool(F.relu(self.conv4(x))))

        x = torch.flatten(x,1) # flatten all dimensions except batch
        #print(x)
        genre = torch.sigmoid(self.fc1(x))
        ins = torch.sigmoid(self.fc2(x))
        key = F.relu(self.fc3(x)).flatten() #cuz key input is 1d
        #print(ins)
        return genre,ins,key


