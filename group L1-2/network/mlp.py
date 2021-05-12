import torch.nn as nn
import torch.nn.functional as F

class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        return F.softmax(self.fc3(out),dim=1)

# model = mlp().cuda()
# print(model)