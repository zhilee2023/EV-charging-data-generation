import torch
import torch.distributions as D


class Beta(torch.nn.Module):
    def __init__(self, D_in, H1,layer,loc_embedding,dim,device):
        super(Beta, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1).to(device)
        self.layers=[torch.nn.Linear(H1, H1).to(device)  for _ in range(layer-1)]
        self.linear3= torch.nn.Linear(H1, 2).to(device)
        self.loc_embedding=torch.nn.Embedding(252, loc_embedding).to(device)
        self.batchnorm1=torch.nn.BatchNorm1d(H1,).to(device)
        self.batchnorms=[torch.nn.BatchNorm1d(H1,).to(device) for _ in range(layer-1)]
        self.softplus=torch.nn.Softplus(beta=1, threshold=20).to(device)
        self.dim=dim
    
    def forward(self, x):
        y=x[:,self.dim]
        x=x[:,torch.arange(x.size(1))!=self.dim]
        x=torch.concat([x[:,0:-1],self.loc_embedding(x[:,-1].to(torch.int))],axis=1)
        x = torch.relu(self.batchnorm1(self.linear1(x)))
        for i in range(len(self.layers)):
            x=torch.relu(self.batchnorms[i](self.layers[i](x)))
        x=self.linear3(x)
        k= self.softplus(x[:,0])
        theta=self.softplus(x[:,1])
        k=torch.clamp(k,min=1.19e-07)
        theta=torch.clamp(theta,min=1.19e-07)
        distribution = torch.distributions.beta.Beta(k,theta)
        loss=-distribution.log_prob(torch.clamp(y,min=1.19e-07,max=1-1.19e-07)).mean()
        return distribution,loss
    
class GMM(torch.nn.Module):
    def __init__(self, D_in, H1, layer, n_comp,loc_embedding,dim,device):
        super(GMM, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1).to(device)       
        self.layers=[torch.nn.Linear(H1, H1).to(device)  for _ in range(layer-1)]
        self.linear3= torch.nn.Linear(H1, 3*n_comp).to(device)
        self.loc_embedding=torch.nn.Embedding(252, loc_embedding).to(device)
        self.batchnorm1=torch.nn.BatchNorm1d(H1,).to(device)
        self.batchnorms=[torch.nn.BatchNorm1d(H1,).to(device) for _ in range(layer-1)]
        self.softplus=torch.nn.Softplus(beta=1, threshold=20).to(device)
        self.dim=dim
        self.n_comp=n_comp
    def forward(self, x):
        y=x[:,self.dim]
        x=x[:,torch.arange(x.size(1))!=self.dim]
        x=torch.concat([x[:,0:-1],self.loc_embedding(x[:,-1].to(torch.int))],axis=1)
        x = torch.relu(self.batchnorm1(self.linear1(x)))
        for i in range(len(self.layers)):
            x=torch.relu(self.batchnorms[i](self.layers[i](x)))
        output=self.linear3(x)
        mus= self.softplus(output[:,:self.n_comp])
        sigs=self.softplus(output[:,self.n_comp:2*self.n_comp])
        weights=torch.softmax(output[:,2*self.n_comp:3*self.n_comp],axis=1)
        mix = D.Categorical(weights)
        comp = D.Independent(D.Normal(mus.unsqueeze(-1),sigs.unsqueeze(-1)), 1)
        gmm = D.MixtureSameFamily(mix, comp)
        loss= -gmm.log_prob(y.unsqueeze(1)).mean()
        return gmm,loss
    

class Discrete1(torch.nn.Module):
    def __init__(self, D_in, H1, layer,D_out,loc_embedding,dim,device):
        super(Discrete1, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1).to(device)
        self.layers=[torch.nn.Linear(H1, H1).to(device)  for _ in range(layer-1)]
        self.linear3 = torch.nn.Linear(H1, D_out).to(device)
        self.lossss_cal=torch.nn.CrossEntropyLoss()
        self.loc_embedding=torch.nn.Embedding(252, loc_embedding).to(device)
        self.batchnorm1=torch.nn.BatchNorm1d(H1,).to(device)
        self.batchnorms=[torch.nn.BatchNorm1d(H1,).to(device) for _ in range(layer-1)]
        self.dim=dim
    
    def forward(self, x):
        y=x[:,self.dim]
        x=x[:,torch.arange(x.size(1))!=self.dim]
        x=torch.concat([x[:,0:-1],self.loc_embedding(x[:,-1].to(torch.int))],axis=1)
        x = torch.relu(self.batchnorm1(self.linear1(x)))
        for i in range(len(self.layers)):
            x=torch.relu(self.batchnorms[i](self.layers[i](x)))
        x = self.linear3(x)
        distribution = torch.distributions.Categorical(torch.softmax(x,axis=1))
        loss=self.lossss_cal(x,y.to(torch.int).to(torch.long))
        return distribution,loss
    
class Discrete2(torch.nn.Module):
    def __init__(self, D_in,H1, layer, D_out,dim,device):
        super(Discrete2, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1).to(device)    
        self.layers=[torch.nn.Linear(H1, H1).to(device)  for _ in range(layer-1)]
        self.linear3 = torch.nn.Linear(H1, D_out).to(device)
        self.lossss_cal=torch.nn.CrossEntropyLoss()  
        self.batchnorm1=torch.nn.BatchNorm1d(H1,).to(device)
        self.batchnorms=[torch.nn.BatchNorm1d(H1,).to(device) for _ in range(layer-1)]
        self.dim=dim
    
    def forward(self, x):
        y=x[:,self.dim]
        x=x[:,torch.arange(x.size(1))!=self.dim]
        x = torch.relu(self.batchnorm1(self.linear1(x)))
        for i in range(len(self.layers)):
            x=torch.relu(self.batchnorms[i](self.layers[i](x)))
        x = self.linear3(x)
        distribution = torch.distributions.Categorical(torch.softmax(x,axis=1))
        loss=self.lossss_cal(x,y.to(torch.int).to(torch.long))
        return distribution,loss