import pandas as pd
import numpy as np
import torch
import numpy as np
import torch.optim as optim
from gibbs_method_network import *
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime
import os
import scipy
import torch.distributions as D
from privacy_cal import privacy_loss,calculate_kl_divergence
import os
import json
from sklearn.model_selection import train_test_split



def reindex_start_hour(x):
    x=x*24-6
    if x<0:
        x=x+24
    return x/24


def gibbs_train(layer,hidden_size,n_comp,beta_func):
    parameter="layer_"+str(layer)+"_hidden_size_"+str(hidden_size)+"_n_comp_"+str(n_comp)+"_beta_func_"+str(beta_func)+"/"
    save_file="gibbs_method_new/"+parameter+"/"
    if not os.path.exists(save_file):
        os.makedirs(save_file)

    features=['Initial SOC','Final SOC','Start time','Duration','Battery capacity','User label','Month','Start location']
    fNum=len(features)
    batch_size = 512
    num_epochs = 200
    learning_rate = 0.005
    hidden_size1=32
    embedding_num=6
    f1_num=fNum+embedding_num-2
    f2_num=fNum-1
    loc_num=252

    df=pd.read_csv('samples_best.csv')
    battery_capacity=df['Battery capacity'].unique()
    battery_label=dict(zip(list(battery_capacity),list(range(len(battery_capacity)))))
    df['Start time']=(df['Start time'].apply(reindex_start_hour))
    df['Month']-=1
    battery_label={35.0: 0, 48.3: 1, 25.0: 2, 37.8: 3, 22.0: 4}
    df['Battery capacity']=df['Battery capacity'].apply(lambda x:battery_label[x])

    df=df[features]
    df['Duration']/=24
    dataset=df.to_numpy()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, drop_last= True,pin_memory=torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if beta_func:
        p_ssoc=Beta(f1_num,hidden_size1,layer,embedding_num,0,device)
        p_end_soc=Beta(f1_num,hidden_size1,layer,embedding_num,1,device)
    else:
        p_ssoc=GMM(f1_num,hidden_size1,layer,n_comp,embedding_num,0,device)
        p_end_soc=GMM(f1_num,hidden_size1,layer,n_comp,embedding_num,1,device)
    p_shour=GMM(f1_num,hidden_size1,layer,n_comp,embedding_num,2,device)
    p_duration=GMM(f1_num,hidden_size1,layer,n_comp,embedding_num,3,device)
    p_battery=Discrete1(f1_num,hidden_size1,layer,len(battery_capacity),embedding_num,4,device)
    p_vlabel=Discrete1(f1_num,hidden_size1,layer,2,embedding_num,5,device)
    p_month=Discrete1(f1_num,hidden_size1,layer,12,embedding_num,6,device)
    p_location=Discrete2(f2_num,hidden_size1,layer,loc_num,7,device)
    nets=[p_ssoc,p_end_soc,p_shour,p_duration,p_battery,p_vlabel,p_month,p_location]

    optimizers = [optim.Adam(i.parameters(), lr=learning_rate) for i in nets]
    schedulers = [lr_scheduler.LinearLR(op, start_factor=1.0, end_factor=0.5, total_iters=30)  for op in optimizers]

    train_losses=[]
    for epoch in range(num_epochs):
        avg=[]
        b=0
        for item in dataloader:
            item=item.to(torch.float32).to(device)
            list(map(lambda op:op.zero_grad() ,optimizers))
            loss=list(map(lambda net:net(item)[1],nets))
            #loss=p_shour(item)#[net(item)[1]  for net in nets]
            list(map(lambda l:l.backward() ,loss))
            list(map(lambda op:op.step() ,optimizers))
            avg.append([l.item() for l in loss])
            b+=1
        [scheduler.step() for scheduler in schedulers]
        avg=np.array(avg)
        l=np.mean(avg,0)
        train_losses.append(l)
        print(epoch, l)
    np.save(save_file+'loss.npy',np.array(train_losses))
    print('batch_num: '+str(b))

    model_name=['p_ssoc','p_end_soc','p_shour','p_duration','p_battery_capacity','p_vlabel','p_month','p_location']
    t=datetime.now().strftime('%Y-%m-%d-%H-%M')
    for i in range (len(features)):
        model=nets[i]
        path=save_file+model_name[i]+"_new_beta_"+t+".pt"
        torch.save(model, path)
    
    sample_size=200000

    #for _ in range(batch_num):
    step=200
    samples=np.zeros((0,len(features)))
    t1=datetime.now()
    with torch.no_grad():
        batch_samples=torch.concat([#torch.randint(size=(sample_size,1),high=101),\
                                    #torch.randint(size=(sample_size,1),high=101),\
                                    torch.rand(sample_size,3),\
                                    torch.rand(sample_size,1),\
                                    torch.randint(size=(sample_size,1),high=len(battery_capacity))\
                                    ,torch.randint(size=(sample_size,1),high=2),\
                                    torch.randint(size=(sample_size,1),high=12),\
                                    torch.randint(size=(sample_size,1),high=loc_num)],  dim=1)
        batch_samples=batch_samples.to(device).to(torch.float32)
        for _ in range(step):
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
    #sample_2.append(net.sample(torch.randn(size=(sample_size,2)).to(device)).detach().cpu().numpy())
    
    #sample_test=np.row_stack(sample_2)

    #['Start location','Start time','Initial SOC','Duration','Final SOC']

    sampling=pd.DataFrame(samples,columns=features)
    t2=datetime.now()
    print(sampling.head())
    sfile=save_file+"samples_"+t+".csv"
    
    sampling=sampling[features]
    sampling.to_csv(sfile,index=False)

    sampling[features[4:]]=sampling.iloc[:,4:].astype('int',copy=False)
    df[features[4:]]=df.iloc[:,4:].astype('int',copy=False)
    len_index=[5,2,12,loc_num]
    distribution_wasserstein=[]
    sample_mean=[]
    true_mean=[]
    for i in  range(8):
        sample_mean.append(sampling[features[i]].mean())
        true_mean.append(df[features[i]].mean())
        if i<4:
            distribution_wasserstein.append(scipy.stats.wasserstein_distance(sampling[features[i]],df[features[i]]))
           
            #print(scipy.stats.wasserstein_distance(sampling[i],df[features[i]]))
        if i>=4:
            kl_div=calculate_kl_divergence(df[features[i]].astype('int'),sampling[features[i]].astype('int'),len_index[i-4])
            distribution_wasserstein.append(kl_div)
    
    result_summary=dict(zip(features,distribution_wasserstein))
    result_summary['privacy_loss']=privacy_loss(train_dataset,sampling.to_numpy(),test_dataset)
    result_summary['sample_mean_diff'] = dict(zip(features, [float(v) for v in sample_mean]))
    result_summary['data_diff'] = dict(zip(features, [float(v) for v in true_mean]))
    result_summary['distribution_wasserstein']=np.sum(distribution_wasserstein)
    result_summary['layer']=layer
    result_summary['hidden_size']=hidden_size
    result_summary['n_comp']=n_comp
    result_summary['sample_time']=str(t2-t1)
    with open(save_file+"result_sum.json", 'w') as file:
        json.dump(result_summary, file, indent=4)