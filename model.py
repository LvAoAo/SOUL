import torch
import torch.nn as nn
import torch.nn.functional as F

#
# class WordleBP(nn.Module):
#     def __init__(self, embedding, embedding_dim, attr_dim, hidden_dim):
#         super(WordleBP, self).__init__()
#         self.embedding = embedding
#         self.embedding_dim = embedding_dim
#         self.attr_dim = attr_dim
#         # self.fc11 = nn.Linear(self.embedding_dim, hidden_dim) # 1-7 tries
#         # self.fc12 = nn.Linear(self.attr_dim, hidden_dim) # 1-7 tries
#         # self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)  # 1-7 tries
#         self.fc3 = nn.Linear(hidden_dim * 2, 7)
#         # self.lasso = LassoRegression(hidden_dim * 2, 7, 0.01)
#         self.fc = nn.Linear(attr_dim + embedding_dim, hidden_dim * 2)
#         # self.ridge = RidgeRegression(hidden_dim * 2, 7, 0.01)
#         self.dropout = nn.Dropout(0.1)
#
#         # self.fc2 = nn.Linear(hidden_dim , hidden_dim)  # 1-7 tries
#
#     def forward(self, indices, attrs):
#         # x is a tuple
#
#         # # print(x)
#         embedding = self.embedding(indices)
#         # embedding = F.relu(self.fc11(embedding))
#         # attrs = F.relu(self.fc12(attrs))
#         x = torch.cat([embedding, attrs], dim=-1)
#         x = F.normalize(x, dim=-1)
#         x = F.relu(self.fc(x))
#         # print(x.shape)
#         x = self.dropout(x)
#         # x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#
#         # _, x = self.ridge(x)
#         # x = F.softmax(x, dim=-1)
#
#         # x = F.normalize(attrs, dim=-1)
#         # x = F.relu(self.fc12(x))
#         # # x = torch.cat([embedding, attr], dim=-1)
#         # # print(x.shape)
#         # x = self.dropout(x)
#         # x = F.relu(self.fc2(x))
#         # x = F.softmax(x, dim=-1)
#         # # print(l1_loss, x.shape)
#         return x

class WordleBP(nn.Module):
    def __init__(self, embedding, embedding_dim, attr_dim, hidden_dim):
        super(WordleBP, self).__init__()
        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.attr_dim = attr_dim
        self.fc1 = nn.Linear(attr_dim + embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 7)
        self.dropout = nn.Dropout(0.1)

    def forward(self, indices, attrs):
        embedding = self.embedding(indices)
        x = torch.cat([embedding, attrs], dim=-1)
        x = F.normalize(x, dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        if self.training:
            x = torch.log(x)


        # print(x)
        return x




class LassoRegression(nn.Module):
    def __init__(self, input_size, output_size, l1_penalty):
        super(LassoRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.l1_penalty = l1_penalty

    def forward(self, x):
        out = self.linear(x)
        l1_loss = self.l1_penalty * torch.norm(self.linear.weight, p=1)
        return l1_loss, out


class RidgeRegression(nn.Module):
    def __init__(self, input_size, output_size, alpha):
        super(RidgeRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.alpha = alpha

    def forward(self, x):
        out = self.linear(x)
        ridge_loss = self.alpha * torch.sum(torch.square(self.linear.weight))
        return ridge_loss, out
