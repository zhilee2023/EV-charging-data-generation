import pandas as pd
import numpy as np
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset_pre import *
from Network import *
import torch.optim.lr_scheduler as lr_scheduler
from torchsummary import summary
from datetime import datetime


features=['start_soc','end_soc','start_hour','duration','battery_capacity','veh_label','month','start_index']
fNum=len(features)
batch_size = 512
num_epochs = 200
learning_rate = 0.005
hidden_size1=32
hidden_size2=32
embedding_num=6
f1_num=fNum+embedding_num-2
f2_num=fNum-1
loc_num=252


p_ssoc=Beta(f1_num,hidden_size1,hidden_size2,1,embedding_num,0).to(device)
p_end_soc=Beta(f1_num,hidden_size1,hidden_size2,1,embedding_num,1).to(device)
p_shour=GMM(f1_num,hidden_size1,hidden_size2,6,embedding_num,2).to(device)
p_duration=GMM(f1_num,hidden_size1,hidden_size2,2,embedding_num,3).to(device)
p_battery=Discrete1(f1_num,hidden_size1,hidden_size2,len(battery_capacity),embedding_num,4).to(device)
p_vlabel=Discrete1(f1_num,hidden_size1,hidden_size2,2,embedding_num,5).to(device)
p_month=Discrete1(f1_num,hidden_size1,hidden_size2,12,embedding_num,6).to(device)
p_location=Discrete2(f2_num,hidden_size1,hidden_size2,loc_num,7).to(device)
nets=[p_ssoc,p_end_soc,p_shour,p_duration,p_battery,p_vlabel,p_month,p_location]
optimizers = [optim.Adam(i.parameters(), lr=learning_rate) for i in nets]
schedulers = [lr_scheduler.LinearLR(op, start_factor=1.0, end_factor=0.5, total_iters=50)  for op in optimizers]


train_losses=[]
for epoch in range(num_epochs):
    avg=[]
    b=0
    for item in dataloader:
        item=item.to(torch.float32).to(device)
        [op.zero_grad()  for op in optimizers]
        loss=[net(item)[1]  for net in nets]
        [l.backward()   for l in loss]
        [op.step() for op in optimizers]
        avg.append([l.item() for l in loss])
        b+=1
    [scheduler.step() for scheduler in schedulers]
    avg=np.array(avg)
    l=np.mean(avg,0)
    train_losses.append(l)
    print(epoch, l)
np.save('loss.npy',np.array(train_losses))
print('batch_num: '+str(b))

model_name=['p_ssoc','p_end_soc','p_shour','p_duration','p_battery_capacity','p_vlabel','p_month','p_location']
t=datetime.now().strftime('%Y-%m-%d-%H-%M')
for i in range (len(features)):
    model=nets[i]
    path="models/"+model_name[i]+"_new_beta_"+t+".pt"
    torch.save(model, path)
