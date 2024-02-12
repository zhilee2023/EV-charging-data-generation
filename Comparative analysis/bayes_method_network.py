import torch
import torch.distributions as D

#'veh_label','battery_capacity','month','start index','start_soc','start_hour','duration'，'end_soc'
class continous(torch.nn.Module):
    def __init__(self, D_in, H1,layer, n_comp,cond_index,output_index,loc_embedding,beta=True ,device='cpu'):
        super(continous, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1).to(device)
        self.layers=[torch.nn.Linear(H1, H1).to(device)  for _ in range(layer-1)]
        self.linear3= torch.nn.Linear(H1, 3*n_comp).to(device)
        self.linear_beta= torch.nn.Linear(H1, 2).to(device)
        self.n_comp=n_comp
        self.cond_index=torch.tensor(cond_index).to(device)
        self.output_index=output_index
        self.batchnorm1=torch.nn.BatchNorm1d(H1,).to(device)
        self.batchnorms=[torch.nn.BatchNorm1d(H1,).to(device) for _ in range(layer-1)]
        #self.batchnorm2=torch.nn.BatchNorm1d(H1,).to(device)
        self.loc_embedding=torch.nn.Embedding(252, loc_embedding).to(device)
        self.softplus=torch.nn.Softplus(beta=1, threshold=20).to(device)
        self.beta=beta
    
    def forward(self,x):
        y=x[:,self.output_index]
        x=x[:,self.cond_index]
        if self.output_index>3:
            x=torch.concat([x[:,0:3],self.loc_embedding(x[:,3].to(torch.int)),x[:,4:self.output_index]],axis=1)
        x = torch.relu(self.batchnorm1(self.linear1(x)))
        for i in range(len(self.layers)-1):
            x=torch.relu(self.batchnorms[i](self.layers[i](x)))
    
        if not self.beta:
            output = self.linear3(x)

            mus= self.softplus(output[:,:self.n_comp])
            sigs=self.softplus(output[:,self.n_comp:2*self.n_comp])
            weights=torch.softmax(output[:,2*self.n_comp:3*self.n_comp],axis=1)
            mix = D.Categorical(weights)
            comp =D.Independent(D.Normal(mus.unsqueeze(-1),sigs.unsqueeze(-1)), 1)
            distribution = D.MixtureSameFamily(mix, comp)
            loss= -distribution.log_prob(y.unsqueeze(1)).mean()

        else:
            output = self.linear_beta(x)
            k= self.softplus(output[:,0])
            theta=self.softplus(output[:,1])
            k=torch.clamp(k,min=1.19e-07)
            theta=torch.clamp(theta,min=1.19e-07)
            distribution = torch.distributions.beta.Beta(k,theta)
            loss= -distribution.log_prob(torch.clamp(y,min=1.19e-07,max=1-1.19e-07)).mean()
        return distribution,loss

class discrete(torch.nn.Module):
    def __init__(self, D_in, H1,layer, num_class,cond_index,output_index,device='cpu'):
        super(discrete, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1).to(device)
        self.layers=[torch.nn.Linear(H1, H1).to(device)  for _ in range(layer-1)]
        self.linear3= torch.nn.Linear(H1,num_class).to(device)
        self.num_class=num_class
        self.batchnorm1=torch.nn.BatchNorm1d(H1,).to(device)
        self.cond_index=torch.tensor(cond_index).to(device)
        self.output_index=output_index
        self.batchnorms=[torch.nn.BatchNorm1d(H1,).to(device) for i in range(layer-1)]
        self.lossss_cal=torch.nn.CrossEntropyLoss()  
    def forward(self,x):
        y=x[:,self.output_index]
        x=x[:,self.cond_index]
        if self.output_index>3:
            x=torch.concat([x[:,0:3],self.loc_embedding(x[:,3].to(torch.int)),x[:,4:self.output_index]],axis=1)
        #x=torch.concat([x[:,0:-1],self.loc_embedding(x[:,-1].to(torch.int))],axis=1)
        x = torch.relu(self.batchnorm1(self.linear1(x)))
        for i in range(len(self.layers)):
            x=torch.relu(self.batchnorms[i](self.layers[i](x)))
        x = self.linear3(x)
        distribution = torch.distributions.Categorical(torch.softmax(x,axis=1))
        loss=self.lossss_cal(x,y.to(torch.long))
        return distribution,loss