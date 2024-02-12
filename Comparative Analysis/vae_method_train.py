import pandas as pd
import numpy as np
import torch
import scipy
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime
from vae_network import VAE
import torch.distributions as D
from privacy_cal import privacy_loss,calculate_kl_divergence
import os
import json
from sklearn.model_selection import train_test_split
from mmd_loss import MMD_loss


def reindex_start_hour(x):
    x=x*24-6
    if x<0:
        x=x+24
    return x/24
#parameters_change



def vae_train(layer,hidden_size,n_comp,z_dim,beta,beta_func):
    parameter="layer_"+str(layer)+"_hidden_size_"+str(hidden_size)+"_n_comp_"+str(n_comp)+\
        "_z_dim_"+str(z_dim)+"_beta_"+str(beta)+"_beta_func_"+str(beta_func)+"/"
    save_file="vae_method_new/"+parameter+"/"
    if not os.path.exists(save_file):
        os.makedirs(save_file)
    fNum=8
    batch_size = 512
    num_epochs = 200
    learning_rate = 0.005
    embedding_num=6
    f1_num=fNum+embedding_num-2
    f2_num=fNum-1
    loc_num=252
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device=torch.device('cpu')
    ##read,process, and create dataset
    df=pd.read_csv('samples_best.csv')
    battery_label={35.0: 0, 48.3: 1, 25.0: 2, 37.8: 3, 22.0: 4}
    df['Start time']=((df['Start time']).apply(reindex_start_hour)) ## reset start time at 6:00 am
    df['Month']-=1
    df['Battery capacity']=df['Battery capacity'].apply(lambda x:battery_label[x])
    df['Duration']/=df['Duration'].max()
    dataset=df.to_numpy()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size)

    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8,0.2])
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, drop_last= True,pin_memory=torch.cuda.is_available())
    
    

    net=VAE(embedding_dim=embedding_num,D_in=f1_num+1, h_dim=hidden_size,n_comp=3,z_dim=2,beta_func=beta_func,device=device)
    ## define optimizer and scheduler
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.LinearLR(optimizer , start_factor=1.0, end_factor=0.5, total_iters=50)


    ## Train the model
    train_losses=[]
    for epoch in range(num_epochs):
        avg=[]
        b=0
        for item in dataloader:
            item=item.to(torch.float32).to(device)
            #[op.zero_grad()  for op in optimizers]
            optimizer.zero_grad()
            kl_loss,recon_loss=net(item)
            loss=beta*kl_loss+recon_loss
            loss.backward()
            optimizer.step()
            
            avg.append([kl_loss.item(),recon_loss.item()])
            b+=1
            if b>1500:
                break;
        scheduler.step()
        avg=np.array(avg)
        l=np.mean(avg,0)
        train_losses.append(l)
        print(epoch, l)
    
    np.save(save_file+'loss.npy',np.array(train_losses))
    #print('batch_num: '+str(b))
    t=datetime.now().strftime('%Y-%m-%d-%H-%M')
    path=save_file+"vae_new_beta_"+t+".pt"
    torch.save(net, path)
    #sampling
    sample_size=200000
    #batch_num=10
    sample_1=[]
    t1=datetime.now()
    #for _ in range(batch_num):
    sample_1.append(net.sample(torch.randn(size=(sample_size,2)).to(device)).detach().cpu().numpy())

    
    samples=np.row_stack(sample_1)

    sampling=pd.DataFrame(samples)
    t2=datetime.now()

    sfile=save_file+"samples_"+t+".csv"
    sampling.to_csv(sfile,index=False)

    
    features=['Initial SOC','Final SOC','Start time','Duration','Battery capacity','User label','Month','Start location']
# 根据features列表重新排序列

    sampling[[4,5,6,7]]=sampling.iloc[:,4:].astype('int',copy=False)
    df[features[4:]]=df.iloc[:,4:].astype('int',copy=False)
    len_index=[5,2,12,loc_num]
    distribution_wasserstein=[]
    distribution_mmd_01=[] #mmd 0.1
    distribution_mmd_05=[] #mmd 0.5
    distribution_mmd_10=[] #mmd 1
    distribution_mmd_50=[] #mmd 5
    sample_mean=[]
    true_mean=[]
    mmd_loss=MMD_loss()
    for i in  range(8):
        sample_mean.append(sampling[i].mean())
        true_mean.append(df[features[i]].mean())
        if i<4:
            distribution_wasserstein.append(scipy.stats.wasserstein_distance(sampling[i],df[features[i]]))
            distribution_mmd_01.append(mmd_loss.multi_trunk_mmd_loss(sampling[i],df[features[i]],sigma=0.1))
            distribution_mmd_05.append(mmd_loss.multi_trunk_mmd_loss(sampling[i],df[features[i]],sigma=0.5))
            distribution_mmd_10.append(mmd_loss.multi_trunk_mmd_loss(sampling[i],df[features[i]],sigma=1))
            distribution_mmd_50.append(mmd_loss.multi_trunk_mmd_loss(sampling[i],df[features[i]],sigma=5))
            #print(scipy.stats.wasserstein_distance(sampling[i],df[features[i]]))
        if i>=4:
            kl_div=calculate_kl_divergence(df[features[i]].astype('int'),sampling[i].astype('int'),len_index[i-4])
            distribution_wasserstein.append(kl_div)
            distribution_mmd_01.append(kl_div)
            distribution_mmd_05.append(kl_div)
            distribution_mmd_10.append(kl_div)
            distribution_mmd_50.append(kl_div)
    
    result_summary=dict(zip(features,distribution_wasserstein))
    result_summary['privacy_loss']=privacy_loss(train_dataset,samples,test_dataset)
    result_summary['sample_mean_diff'] = dict(zip(features, [float(v) for v in sample_mean]))
    result_summary['data_diff'] = dict(zip(features, [float(v) for v in true_mean]))
    result_summary['distribution_wasserstein']=np.sum(distribution_wasserstein)
    result_summary['distribution_mmd_01']=np.sum(distribution_mmd_01)
    result_summary['distribution_mmd_05']=np.sum(distribution_mmd_05)
    result_summary['distribution_mmd_10']=np.sum(distribution_mmd_10)
    result_summary['distribution_mmd_50']=np.sum(distribution_mmd_50)

    result_summary['layer']=layer
    result_summary['hidden_size']=hidden_size
    result_summary['n_comp']=n_comp
    result_summary['z_dim']=z_dim
    result_summary['beta']=beta
    result_summary['sample_time']=str(t2-t1)
 
    with open(save_file+"result_sum.json", 'w') as file:
        json.dump(result_summary, file, indent=4)