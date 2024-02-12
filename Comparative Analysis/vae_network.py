import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
class VAE(nn.Module):
    def __init__(self, embedding_dim,D_in, h_dim,n_comp,z_dim,layer=1,beta_func=True,device='cpu'):
        super(VAE, self).__init__()
        #['start_soc','end_soc','start_hour','duration','battery_capacity','veh_label','month','start_index']
        self.embed=torch.nn.Embedding(252, embedding_dim).to(device)
        self.encoder_fc1 =nn.Linear(D_in, h_dim).to(device)
        self.encoder_layers= [torch.nn.Linear(h_dim, h_dim).to(device)  for _ in range(layer-1)]
        self.zoutput = nn.Linear(h_dim, z_dim*2).to(device)

        self.decoder_fc1 =nn.Linear(z_dim, h_dim).to(device)
        self.decoder_layers =[torch.nn.Linear(h_dim, h_dim).to(device)  for _ in range(layer-1)]
        self.beta_func=beta_func
        #self.decoder_continuous=nn.Linear(h_dim, 4).to(device)
        if beta_func:
            self.decoder_start_soc=nn.Linear(h_dim, 2).to(device)
            self.decoder_end_soc=nn.Linear(h_dim, 2).to(device)
        else:    
            self.decoder_start_soc=nn.Linear(h_dim, n_comp*3).to(device)
            self.decoder_end_soc=nn.Linear(h_dim, n_comp*3).to(device)
        self.decoder_start_hour=nn.Linear(h_dim, 3*n_comp).to(device)
        self.decoder_duration=nn.Linear(h_dim, 3*n_comp).to(device)
        self.decoder_battery_capacity=nn.Linear(h_dim, 5).to(device)
        self.decoder_veh_label=nn.Linear(h_dim, 2).to(device)
        self.decoder_month=nn.Linear(h_dim, 12).to(device)
        self.decoder_loc=nn.Linear(h_dim, 252).to(device)
        self.n_comp=n_comp
        self.layer=layer
        self.encoder_batchnorm1=torch.nn.BatchNorm1d(h_dim,).to(device)
        self.decoder_batchnorm1=torch.nn.BatchNorm1d(h_dim,).to(device)
        self.encoder_batchnorms=[torch.nn.BatchNorm1d(h_dim,).to(device) for _ in range(layer-1)]
        self.decoder_batchnorms=[torch.nn.BatchNorm1d(h_dim,).to(device) for _ in range(layer-1)]
        self.device=device
        self.softplus=torch.nn.Softplus(beta=1, threshold=20)
        
        self.lossss_cal=torch.nn.CrossEntropyLoss()
        self.mse=torch.nn.MSELoss()
        self.z_dim=z_dim
    def encoder(self, x):
        x= torch.relu(self.encoder_batchnorm1(self.encoder_fc1(x)))
        for i in range(len(self.encoder_layers)):
            x=torch.relu(self.encoder_batchnorms[i](self.encoder_layers[i](x)))
        z=self.zoutput(x)
        return z

    def reparameterize(self, z):
        mu=z[:,:self.z_dim]
        logvar=z[:,self.z_dim:]
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps*std
    #['start_soc','end_soc','start_hour','duration','battery_capacity','veh_label','month','start_index']
    def decoder(self, z):
        x= torch.relu(self.decoder_batchnorm1(self.decoder_fc1(z)))
        for i in range(len(self.decoder_layers)):
            x=torch.relu(self.decoder_batchnorms[i](self.decoder_layers[i](x)))
        return x

    def kl_div_loss(self,z):
        mu=z[:,:self.z_dim]
        logvar=z[:,self.z_dim:]
        #epsilon = 1e-8
        kl_div_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))
        return kl_div_loss
    
    def gmm(self, input):
        mus= self.softplus(input[:,:self.n_comp])
        sigs=self.softplus(input[:,self.n_comp:2*self.n_comp])
        weights=torch.softmax(input[:,2*self.n_comp:3*self.n_comp],axis=1)
        mix = D.Categorical(weights)
        comp =D.Independent(D.Normal(mus.unsqueeze(-1),sigs.unsqueeze(-1)), 1)
        gmm = D.MixtureSameFamily(mix, comp)
        return gmm
    
    def beta(self,input):

        k= self.softplus(input[:,0])
        theta=self.softplus(input[:,1])
        k=torch.clamp(k,min=1.19e-07)
        theta=torch.clamp(theta,min=1.19e-07)
        distribution = torch.distributions.beta.Beta(k,theta)
        return distribution

    def recon_loss(self,h,y):
        #recon_con=torch.sigmoid(self.decoder_continuous(h))
        #recon_con=torch.cat([torch.sigmoid(recon[:,:3]),torch.relu()])
        #continuous_loss=self.mse(self.decoder_continuous(h),y[:,:4])
                        #y=torch.clamp(y,min=1.19e-07,max=1-1.19e-07)
                #loss=-distribution.log_prob(y.unsqueeze(1)).mean()
        if self.beta_func:
            start_soc=-self.beta(self.decoder_start_soc(h)).log_prob(torch.clamp(y[:,0:1],min=1.19e-07,max=1-1.19e-07)).mean()
            end_soc=-self.beta(self.decoder_end_soc(h)).log_prob(torch.clamp(y[:,1:2],min=1.19e-07,max=1-1.19e-07)).mean()
        else:
            start_soc=-self.gmm(self.decoder_start_soc(h)).log_prob(y[:,0:1]).mean()
            end_soc=-self.gmm(self.decoder_end_soc(h)).log_prob(y[:,1:2]).mean()
        start_hour=-self.gmm(self.decoder_start_hour(h)).log_prob(y[:,2:3]).mean()
        duration=-self.gmm(self.decoder_duration(h)).log_prob(y[:,3:4]).mean()
        #self.decoder_battery_capacity=nn.Linear(h_dim, 5)
        battery_capacity_loss=self.lossss_cal(self.decoder_battery_capacity(h),y[:,4].to(torch.long ))
        veh_label_loss=self.lossss_cal(self.decoder_veh_label(h),y[:,5].to(torch.long ))
        month_loss=self.lossss_cal(self.decoder_month(h),y[:,6].to(torch.long ))
        loc_loss=self.lossss_cal(self.decoder_loc(h),y[:,7].to(torch.long))
        recon_loss=start_soc+end_soc+start_hour+duration+(battery_capacity_loss+veh_label_loss+ month_loss+loc_loss)

        return recon_loss

    def forward(self, x):
        y=x.clone().detach()
        x=torch.concat([x[:,0:-1],self.embed(x[:,-1].to(torch.int))],axis=1)
        z = self.encoder(x)
        kl_loss=self.kl_div_loss(z)
        z = self.reparameterize(z)
        h=self.decoder(z)
        recon_loss= self.recon_loss(h,y)
        #loss=recon_loss+kl_loss
        return kl_loss,recon_loss

    def sample(self,z):
        with torch.no_grad():
            h=self.decoder(z)
            if self.beta_func:
                start_soc=self.beta(self.decoder_start_soc(h)).sample().unsqueeze(1)
                end_soc=self.beta(self.decoder_end_soc(h)).sample().unsqueeze(1)
            else:        
                start_soc=self.gmm(self.decoder_start_soc(h)).sample()
                end_soc=self.gmm(self.decoder_end_soc(h)).sample()
            start_hour=self.gmm(self.decoder_start_hour(h)).sample()
            duration=self.gmm(self.decoder_duration(h)).sample()
            battery_capacity = torch.distributions.Categorical(torch.softmax(self.decoder_battery_capacity(h),axis=1)).sample().unsqueeze(1)
            veh_label= torch.distributions.Categorical(torch.softmax(self.decoder_veh_label(h),axis=1)).sample().unsqueeze(1)
            month= torch.distributions.Categorical(torch.softmax(self.decoder_month(h),axis=1)).sample().unsqueeze(1)
            loc= torch.distributions.Categorical(torch.softmax(self.decoder_loc(h),axis=1)).sample().unsqueeze(1)
            samples=torch.cat([start_soc,end_soc,start_hour,duration,battery_capacity,veh_label,month,loc],dim=-1)
        return samples

