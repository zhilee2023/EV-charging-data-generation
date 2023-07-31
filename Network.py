import torch
import torch.distributions as D


class Beta(torch.nn.Module):
    def __init__(self, D_in, H1,H2, D_out,loc_embedding,dim):
        super(Beta, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear_para1= torch.nn.Linear(H2, D_out)
        self.linear_para2= torch.nn.Linear(H2, D_out)
        self.loc_embedding=torch.nn.Embedding(252, loc_embedding)
        self.batchnorm1=torch.nn.BatchNorm1d(H1,)
        self.batchnorm2=torch.nn.BatchNorm1d(H2,)
        self.softplus=torch.nn.Softplus(beta=1, threshold=20)
        self.dim=dim
    
    def forward(self, x):
        y=x[:,self.dim]
        x=x[:,torch.arange(x.size(1))!=self.dim]
        x=torch.concat([x[:,0:-1],self.loc_embedding(x[:,-1].to(torch.int))],axis=1)
        x = torch.relu(self.batchnorm1(self.linear1(x)))
        x = torch.relu(self.batchnorm2(self.linear2(x)))
        k= self.softplus(self.linear_para1(x))
        theta=self.softplus(self.linear_para2(x))
        k=torch.clamp(k,min=1.19e-07)
        theta=torch.clamp(theta,min=1.19e-07)
        distribution = torch.distributions.beta.Beta(k,theta)
        y=torch.clamp(y,min=1.19e-07,max=1-1.19e-07)
        loss=-distribution.log_prob(y.unsqueeze(1)).mean()
        return distribution,loss
    

class GMM(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out,loc_embedding,dim):
        super(GMM, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)       
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear_para1= torch.nn.Linear(H2, D_out)
        self.linear_para2= torch.nn.Linear(H2, D_out)
        self.linear_para3= torch.nn.Linear(H2, D_out)
        self.loc_embedding=torch.nn.Embedding(252, loc_embedding)
        self.batchnorm1=torch.nn.BatchNorm1d(H1,)
        self.batchnorm2=torch.nn.BatchNorm1d(H2,)
        self.softplus=torch.nn.Softplus(beta=1, threshold=20)
        self.dim=dim
    
    def forward(self, x):
        y=x[:,self.dim]
        x=x[:,torch.arange(x.size(1))!=self.dim]
        x=torch.concat([x[:,0:-1],self.loc_embedding(x[:,-1].to(torch.int))],axis=1)
        x = torch.relu(self.batchnorm1(self.linear1(x)))
        x = torch.relu(self.batchnorm2(self.linear2(x)))
        mus= self.softplus(self.linear_para1(x))
        sigs=self.softplus(self.linear_para2(x))
        weights=torch.softmax(self.linear_para3(x),axis=1)
        mix = D.Categorical(weights)
        comp = D.Independent(D.Normal(mus.unsqueeze(-1),sigs.unsqueeze(-1)), 1)
        gmm = D.MixtureSameFamily(mix, comp)
        loss= -gmm.log_prob(y.unsqueeze(1)).mean()
        return gmm,loss
    

class Discrete1(torch.nn.Module):
    def __init__(self, D_in, H1, H2,D_out,loc_embedding,dim):
        super(Discrete1, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)       
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)
        self.lossss_cal=torch.nn.CrossEntropyLoss()
        self.loc_embedding=torch.nn.Embedding(252, loc_embedding)
        self.batchnorm1=torch.nn.BatchNorm1d(H1,)
        self.batchnorm2=torch.nn.BatchNorm1d(H2,)
        self.dim=dim
    
    def forward(self, x):
        y=x[:,self.dim]
        x=x[:,torch.arange(x.size(1))!=self.dim]
        x=torch.concat([x[:,0:-1],self.loc_embedding(x[:,-1].to(torch.int))],axis=1)
        x = torch.relu(self.batchnorm1(self.linear1(x)))
        x = torch.relu(self.batchnorm2(self.linear2(x)))
        x = self.linear3(x)
        distribution = torch.distributions.Categorical(torch.softmax(x,axis=1))
        loss=self.lossss_cal(x,y.to(torch.int).to(torch.long))
        return distribution,loss
    
class Discrete2(torch.nn.Module):
    def __init__(self, D_in,H1, H2, D_out,dim):
        super(Discrete2, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)       
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)
        self.lossss_cal=torch.nn.CrossEntropyLoss()  
        self.batchnorm1=torch.nn.BatchNorm1d(H1,)
        self.batchnorm2=torch.nn.BatchNorm1d(H2,)
        self.dim=dim
    
    def forward(self, x):
        y=x[:,self.dim]
        x=x[:,torch.arange(x.size(1))!=self.dim]
        x = torch.relu(self.batchnorm1(self.linear1(x)))
        x = torch.relu(self.batchnorm2(self.linear2(x)))
        x = self.linear3(x)
        distribution = torch.distributions.Categorical(torch.softmax(x,axis=1))
        loss=self.lossss_cal(x,y.to(torch.int).to(torch.long))
        return distribution,loss