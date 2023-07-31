import pandas as pd
import numpy as np
import torch
from util import *
from Network import *


## set network parameters
features=['start_soc','end_soc','start_hour','duration','battery_capacity','veh_label','month','start_index']
fNum=len(features)
hidden_size1=32
hidden_size2=32
embedding_num=6
f1_num=fNum+embedding_num-2
f2_num=fNum-1
loc_num=252
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
p_ssoc=Beta(f1_num,hidden_size1,hidden_size2,1,embedding_num,0).to(device)
p_end_soc=Beta(f1_num,hidden_size1,hidden_size2,1,embedding_num,1).to(device)
p_shour=GMM(f1_num,hidden_size1,hidden_size2,6,embedding_num,2).to(device)
p_duration=GMM(f1_num,hidden_size1,hidden_size2,2,embedding_num,3).to(device)
p_battery=Discrete1(f1_num,hidden_size1,hidden_size2,5,embedding_num,4).to(device)
p_vlabel=Discrete1(f1_num,hidden_size1,hidden_size2,2,embedding_num,5).to(device)
p_month=Discrete1(f1_num,hidden_size1,hidden_size2,12,embedding_num,6).to(device)
p_location=Discrete2(f2_num,hidden_size1,hidden_size2,loc_num,7).to(device)
nets=[p_ssoc,p_end_soc,p_shour,p_duration,p_battery,p_vlabel,p_month,p_location]


##load trained models
model_name=['p_ssoc','p_end_soc','p_shour','p_duration','p_battery_capacity','p_vlabel','p_month','p_location']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i in range (len(model_name)):
    path="models/"+model_name[i]+".pth"
    nets[i].load_state_dict(torch.load(path))


## set batch size and batch num
sample_size=2048*32
step=100
batch_num=25


##initialize the samples
samples=np.zeros((0,len(features)))
## Gibbs sampling
with torch.no_grad():
    for i in range(batch_num):
        batch_samples=torch.concat([#torch.randint(size=(sample_size,1),high=101),\
                                    #torch.randint(size=(sample_size,1),high=101),\
                                    torch.rand(sample_size,3),\
                                    torch.rand(sample_size,1)*24,\
                                    torch.randint(size=(sample_size,1),high=5)\
                                    ,torch.randint(size=(sample_size,1),high=2),\
                                    torch.randint(size=(sample_size,1),high=12),\
                                    torch.randint(size=(sample_size,1),high=loc_num)],  dim=1)
        batch_samples=batch_samples.to(device).to(torch.float32)
        for s in range(step):
            for i in range(len(features)):
                distribution,_=nets[i].eval()(batch_samples)
                if i <=1:
                    batch_samples[:,i]=distribution.sample().squeeze()
                elif i==2:
                    batch_samples[:,i]=torch.clamp(distribution.sample().squeeze(),0,1)
                elif i==3:
                    batch_samples[:,i]=torch.clamp(distribution.sample().squeeze(),0)
                else:
                    batch_samples[:,i]=distribution.sample().squeeze()
        #print(batch_samples[1].detach().cpu().numpy().mean())
        samples=np.concatenate([samples,batch_samples.detach().cpu().numpy()],axis=0)


## delete some samples and save samples
features=['Initial SOC','Final SOC','Start time','Duration','Battery capacity','User label','Month','Start location']
samples=pd.DataFrame(samples)
sfile="Gen_Samples.csv"
samples=samples.set_axis(features,axis=1)
samples['Month']+=1
samples['Start time']=samples['Start time'].apply(reset_startT_0)
samples['Battery capacity']=samples['Battery capacity'].apply(lambda x:list(battery_label.keys())[int(x)])
samples=samples[(samples['Final SOC']-samples['Initial SOC'])>0]
samples=samples[samples['Duration']!=0]
b=((samples['Final SOC']-samples['Initial SOC'])*samples['Battery capacity']/samples['Duration']/100)
samples=samples[b<120] ##power filtering
samples.to_csv(sfile,index=False)