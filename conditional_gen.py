import pandas as pd
import numpy as np
import torch

features=['start_soc','end_soc','start_hour','duration','battery_capacity','veh_label','month','start_index']
model_name=['p_ssoc','p_end_soc','p_shour','p_duration','p_battery_capacity','p_vlabel','p_month','p_location']
nets=[]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for i in range (len(features)):
    path="models/"+model_name[i]+“.pt"
    nets.append(torch.load(path))

num=10000
day=365
frequency=0.82
total_num=num*365*0.82


sample_size=2048*32
step=200
batch_num=int(total_num//sample_size)
samples=np.zeros((0,len(features)))
total_sample_num=0
with torch.no_grad():
    while total_sample_num<total_num:
        batch_samples=torch.concat([
                                    torch.rand(sample_size,3),\
                                    torch.rand(sample_size,1)*24,\
                                    torch.from_numpy(np.ones((sample_size,1))*4),\
                                    torch.from_numpy(np.zeros((sample_size,1))),\
                                    #torch.randint(size=(sample_size,1),0,2)\
                                    #torch.randint(size=(sample_size,1),high=2),\
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
                elif i==4 or i==5:
                    a=1
                else:
                    batch_samples[:,i]=distribution.sample().squeeze()
        samples=np.concatenate([samples,batch_samples.detach().cpu().numpy()],axis=0)
        total_sample_num=samples.shape[0]
