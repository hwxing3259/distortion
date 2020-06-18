import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas
import scipy.stats
import scipy.special
import matplotlib.pyplot as plt
import statsmodels.api as sm


class MDN(nn.Module):
    def __init__(self, n_hidden, n_beta):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(3, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_beta)
        self.z_alpha = nn.Linear(n_hidden, n_beta)
        self.z_beta = nn.Linear(n_hidden, n_beta)

    def forward(self, x):
        z_h = self.z_h(x)
        pi = F.softmax(self.z_pi(z_h), -1)
        alpha = torch.exp(self.z_alpha(z_h))
        beta = torch.exp(self.z_beta(z_h))
        return pi, alpha, beta



def mdn_loss_fn(p, alpha, beta, pi):
    m = torch.distributions.Beta(alpha, beta)
    loss = torch.exp(m.log_prob(p))
    loss = torch.sum(loss * pi, dim=1)
    loss = -torch.log(loss)
    return torch.mean(loss)




model = MDN(n_hidden=80, n_beta=1) #MDN with one component is sifficient for our problem

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

traindata = pandas.read_csv('ergmabc_train.csv')
# traindata = pandas.read_csv('ergmabc_train.csv')
 # traindata = pandas.read_csv("ergmadjlkd_trainnarm.csv")
traindata = traindata.drop("Unnamed: 0", 1)

traindata = torch.tensor(traindata.values)

x_data = traindata[:, [9, 10, 11]].float()

y_data = traindata[:, 0]
y_data = y_data.view(-1, 1).float()

for epoch in range(10000):
    pi, alpha, beta = model(x_data)
    loss = mdn_loss_fn(y_data, alpha, beta, pi)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(loss.data.tolist())


myfittedbeta = torch.stack(model(torch.tensor([ 78.00000, 73.43855, 63.08138]).float())).detach() #summary statistics of Karate data
print(myfittedbeta)




