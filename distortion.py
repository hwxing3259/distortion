import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas
#import scipy.special
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
from mpl_toolkits.mplot3d import Axes3D


class MDN(nn.Module):
    def __init__(self, inputdim, n_hidden, n_beta):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
            nn.Linear(inputdim, n_hidden),
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




class distortionmap:
    def __init__(self, Y, Q, yobs, n_hidden, n_beta):
        self.y = Y
        self.q = Q
        self.yobs = yobs
        self.nnmodel = MDN(inputdim=self.y.shape[1],n_hidden=n_hidden, n_beta=n_beta)

    def fitmap(self, iterr, mdn_loss_fn):
        optimizer = torch.optim.Adam(self.nnmodel.parameters(), lr=0.0001)
        for epoch in range(iterr):
            pi, alpha, beta = self.nnmodel(self.y)
            loss = mdn_loss_fn(self.q, alpha, beta, pi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                print(loss.data.tolist())

    def fitted(self):
        with torch.no_grad():
            pifit, alphafit, betafit = self.nnmodel(torch.tensor(self.yobs, dtype=torch.float32))
        return pifit.numpy(), alphafit.numpy(), betafit.numpy()





#Here is an example, estimate the distortion map for the adj-lkd posterior of the first parameter of the Karate network example
traindata = pandas.read_csv('ergmadjlkd_trainnarm.csv')
traindata = traindata.drop("Unnamed: 0", 1)

traindata = torch.tensor(traindata.values)

x_data = traindata[:, [9, 10, 11]].float()

y_data = traindata[:, 0]
y_data = y_data.view(-1, 1).float()
yobs=[ 78.00000, 73.43855, 63.08138]

fitdist = distortionmap(x_data,y_data,yobs,80,1)
fitdist.fitmap(7000,mdn_loss_fn)
p1,a1,b1 = fitdist.fitted() #a1,b1 are the fitted parameters of the Beta density, we are using 1 component so p1 is always 1



########################################################################################################################

#bivariate distortion surface
#we give an example of estimating the distortion surface of the the first two dimension (x1,x2) of the adj-lkd posterior

class bivardistortionmap:
    def __init__(self, Y, Q, yobs, n_hidden, n_beta):
        self.y = Y
        self.q = Q
        self.yobs = yobs
        self.nnmodel = MDN(inputdim=self.y.shape[1],n_hidden=n_hidden, n_beta=n_beta)

    def fitmap(self, iterr, mdn_loss_fn):
        optimizer = torch.optim.Adam(self.nnmodel.parameters(), lr=0.0001)
        for epoch in range(iterr):
            pi, alpha, beta = self.nnmodel(self.y)
            loss = mdn_loss_fn(self.q, alpha, beta, pi)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                print(loss.data.tolist())

    def mysampling(self, alpha1, beta1, firstdimsamp, secdimsamp, K):
        oldcdf = sm.nonparametric.KDEUnivariate(firstdimsamp)
        oldcdf.fit()
        testcdf = np.cumsum(oldcdf.density) * (oldcdf.support[1] - oldcdf.support[0])
        newcdf = scipy.stats.beta.cdf(testcdf, a=alpha1, b=beta1)
        newcdf = np.diff(np.append(0, newcdf))
        newsample1 = np.random.choice(oldcdf.support, size=K, replace=True, p=newcdf)
        dens_c = sm.nonparametric.KDEMultivariateConditional(endog=secdimsamp,
                                                             exog=firstdimsamp, dep_type='c', indep_type='c',
                                                             bw='normal_reference')
        newsample2 = np.zeros([K, 1])

        for i in range(K):
            if i % 100 == 0:
                print(i)
            myfittedbeta = torch.stack(self.nnmodel(torch.tensor(np.append(self.yobs, newsample1[i])).float())).detach()
            newalpha = myfittedbeta.numpy()[1]
            newbeta = myfittedbeta.numpy()[2]
            cdesupport = np.linspace(start=np.amin(secdimsamp) - 0.2, stop=np.amax(secdimsamp) + 0.2, num=1000)
            oldcondpdf = dens_c.pdf(endog_predict=cdesupport, exog_predict=np.repeat(newsample1[i], 1000))
            oldcondcdf = np.cumsum(oldcondpdf) * (cdesupport[1] - cdesupport[0])
            newcondcdf = scipy.stats.beta.cdf(oldcondcdf, a=newalpha, b=newbeta)
            newcondcdf = np.diff(np.append(0, newcondcdf))
            newcondcdf = newcondcdf / sum(newcondcdf)
            newsample2[i] = np.random.choice(cdesupport, size=1, replace=True, p=newcondcdf)

        return np.vstack((newsample1, newsample2.T))

    def distortsurface(self, alpha1, beta1, firstdimsamp, K):
        heatmap = np.zeros(K ** 2).reshape(K, K)
        resolution = np.linspace(0.04, 0.96, K)
        dim1quantile = np.quantile(firstdimsamp, resolution)
        distortdim1 = scipy.stats.beta.pdf(resolution, alpha1, beta1)
        for i, dim1samp in enumerate(dim1quantile):
            myfittedbeta = torch.stack(self.nnmodel(torch.tensor(np.append(self.yobs, dim1samp)).float())).detach()
            condalpha = myfittedbeta.numpy()[1]
            condbeta = myfittedbeta.numpy()[2]
            condbetapdf = scipy.stats.beta.pdf(resolution, condalpha, condbeta)
            heatmap[i,] = condbetapdf * distortdim1[i]
        return heatmap

    def plotsurface(self,heatmapsquare):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y = np.meshgrid(np.linspace(0.03, 0.97, 100), np.linspace(0.03, 0.97, 100))
        c = ax.plot_surface(X=x, Y=y, Z=heatmapsquare, cmap=plt.cm.coolwarm, vmin=0.5, vmax=3.1)
        fig.colorbar(c, ax=ax)

        ax.set_xlabel('p_1')
        ax.set_ylabel('p_2')
        ax.set_zlabel('distortion')
        ax.set_zlim(0.5, 3)


#colnames: 1 2 3 1|2, 1|3, 2|1, 2|3, 3|1, 3|2, summ1, summ2, summ3, par1, par2, par3,
#indx:     0 1 2  3    4    5    6    7    8     9      10     11    12    13    14


traindata = pandas.read_csv("ergmadjlkdcond_train.csv")
traindata = traindata.drop("Unnamed: 0",1)
traindata = traindata.drop("X",1)
traindata = torch.tensor(traindata.values)
ergmadjlkdapproxpost = pandas.read_csv("ergmadjlkdpost.csv").values
#want: 1:2

x_data = traindata[:, [9,10,11,12]].float() #the first parameter is now treated as part of the input


y_data = traindata[:, 5]    #the conditional approx posterior X2|X1,y_obs is what we need
y_data = y_data.view(-1, 1).float()

ergmadjlkdapproxpost = pandas.read_csv("ergmadjlkdpost.csv").values
#want: 1:2

bivardist = bivardistortionmap(x_data,y_data,yobs,50,1)
bivardist.fitmap(5000,mdn_loss_fn)
heatmapsquare = bivardist.distortsurface(a1,b1,ergmadjlkdapproxpost[:,1],100)
bivardist.plotsurface(heatmapsquare)


