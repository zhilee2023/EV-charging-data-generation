import pandas as pd
import numpy as np
import torch

model_name=['p_ssoc','p_end_soc','p_shour','p_duration','p_battery_capacity','p_vlabel','p_month','p_location']
nets=[]
for i in range (len(features)):
    path="models/"+model_name[i]+"_.pt"
    nets.load(torch.save(model, path))


sample_size=2048*32
step=100
batch_num=864//32
samples=np.zeros((0,len(features)))
with torch.no_grad():
    for i in range(batch_num):
        batch_samples=torch.concat([#torch.randint(size=(sample_size,1),high=101),\
                                    #torch.randint(size=(sample_size,1),high=101),\
                                    torch.rand(sample_size,3),\
                                    torch.rand(sample_size,1)*24,\
                                    torch.randint(size=(sample_size,1),high=len(battery_capacity))\
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
        samples=np.concatenate([samples,batch_samples.detach().cpu().numpy()],axis=0)

sampling=pd.DataFrame(samples)
sfile="samples/samples.csv"
sampling.to_csv(sfile,index=False)
