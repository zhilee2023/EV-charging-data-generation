import pandas as pd
import numpy as np
import torch
import numpy as np
import torch.optim as optim
from bayes_method_network import *
import torch.optim.lr_scheduler as lr_scheduler
from datetime import datetime
import os
import scipy
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


def bayes_train(layer,hidden_size,n_comp,beta_func):
    parameter="layer_"+str(layer)+"_hidden_size_"+str(hidden_size)+"_n_comp_"+str(n_comp)+"_beta_func_"+str(beta_func)+"/"
    save_file="bayes_method_new/"+parameter+"/"
    if not os.path.exists(save_file):
        os.makedirs(save_file)
    batch_size = 512
    num_epochs = 100
    learning_rate=0.005

    embedding_num=6

    features=['Start time','Initial SOC','Duration','Final SOC']
    fNum=len(features)
    batch_size = 512
    learning_rate = 0.005
    hidden_size1=hidden_size
    embedding_num=6
    loc_num=252

    df=pd.read_csv('samples_best.csv')
    battery_capacity=df['Battery capacity'].unique()
    battery_label=dict(zip(list(battery_capacity),list(range(len(battery_capacity)))))
    df['Start time']=(df['Start time'].apply(reindex_start_hour))
    df['Month']-=1
    battery_label={35.0: 0, 48.3: 1, 25.0: 2, 37.8: 3, 22.0: 4}
    df['Battery capacity']=df['Battery capacity'].apply(lambda x:battery_label[x])
    columns=['User label','Battery capacity','Month','Start location','Start time','Initial SOC','Duration','Final SOC']
    df=df[columns]
    df['Duration']/=24
    dataset=df.to_numpy()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = train_test_split(dataset, test_size=test_size)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True, drop_last= True,pin_memory=torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    p_shour=continous(D_in=3+embedding_num, H1=hidden_size1,layer=layer, n_comp=n_comp,cond_index=[0,1,2,3]\
                  ,loc_embedding=embedding_num,output_index=4,device=device)
    p_ssoc=continous(D_in=4+embedding_num, H1=hidden_size1,layer=layer, n_comp=n_comp,cond_index=[0,1,2,3,4]\
                    ,loc_embedding=embedding_num,output_index=5,device=device)
    p_duration=continous(D_in=5+embedding_num, H1=hidden_size1,layer=layer, n_comp=n_comp,cond_index=[0,1,2,3,4,5]\
                        ,loc_embedding=embedding_num,output_index=6,device=device)
    p_end_soc=continous(D_in=6+embedding_num, H1=hidden_size1,layer=layer, n_comp=n_comp,cond_index=[0,1,2,3,4,5,6]\
                        ,loc_embedding=embedding_num,output_index=7,device=device)


    nets=[p_shour,p_ssoc,p_duration,p_end_soc]
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
            avg.append(np.sum([l.item() for l in loss]))
            b+=1
        [scheduler.step() for scheduler in schedulers]
        avg=np.array(avg)
        l=np.mean(avg)
        train_losses.append(l)
        print(epoch, l)
    np.save(save_file+'loss.npy',np.array(train_losses))
    print('batch_num: '+str(b))

    model_name=['p_shour','p_ssoc','p_duration','p_esoc']
    t=datetime.now().strftime('%Y-%m-%d-%H-%M')
    for i in range (len(features)):
        model=nets[i]
        path=save_file+model_name[i]+"_new_beta_"+t+".pt"
        torch.save(model, path)
    
    sample_size=200000
    t1=datetime.now()
    #for _ in range(batch_num):
    prob=df.groupby(['User label','Battery capacity','Month','Start location']).agg('count')['Start time'].reset_index()
    prob['prob']=prob['Start time']/prob['Start time'].sum()
    prob_feature=prob[['User label','Battery capacity','Month','Start location']].to_numpy()
    #sample_2.append(net.sample(torch.randn(size=(sample_size,2)).to(device)).detach().cpu().numpy())
    
    sampled_features = np.random.choice(range(len(prob)),size=sample_size, p=prob['prob'])
    samples_discrete=np.row_stack(list(map(lambda i:prob_feature[i],sampled_features)))
    samples_discrete=np.column_stack([samples_discrete,np.zeros((sample_size,4))])
    #sample_test=np.row_stack(sample_2)

    #['Start location','Start time','Initial SOC','Duration','Final SOC']
    
    with torch.no_grad():
        f=torch.from_numpy(samples_discrete).to(device).to(torch.float)
        for i in range(len(nets)):
            distribution,_=nets[i].eval()(f)
            feature=distribution.sample().squeeze()
            f[:,i+4]=feature#.to(torch.float)
    
    sampling=pd.DataFrame(f.cpu().numpy(),columns=columns)
    t2=datetime.now()
    sfile=save_file+"samples_"+t+".csv"
    features=['Initial SOC','Final SOC','Start time','Duration','Battery capacity','User label','Month','Start location']
    columns_index = {column: index for index, column in enumerate(columns)}
    sorted_index = [columns_index[feature] for feature in features]
    sampling=sampling[features]
    print(sampling.head())
    sampling.to_csv(sfile,index=False)
    df=df[features]

    sampling[features[4:]]=sampling.iloc[:,4:].astype('int',copy=False)
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
        sample_mean.append(sampling[features[i]].mean())
        true_mean.append(df[features[i]].mean())
        if i<4:
            distribution_wasserstein.append(scipy.stats.wasserstein_distance(sampling[features[i]],df[features[i]]))
            distribution_mmd_01.append(mmd_loss.multi_trunk_mmd_loss(sampling[features[i]],df[features[i]],sigma=0.1))
            distribution_mmd_05.append(mmd_loss.multi_trunk_mmd_loss(sampling[features[i]],df[features[i]],sigma=0.5))
            distribution_mmd_10.append(mmd_loss.multi_trunk_mmd_loss(sampling[features[i]],df[features[i]],sigma=1))
            distribution_mmd_50.append(mmd_loss.multi_trunk_mmd_loss(sampling[features[i]],df[features[i]],sigma=5))
            #print(scipy.stats.wasserstein_distance(sampling[i],df[features[i]]))
        if i>=4:
            kl_div=calculate_kl_divergence(df[features[i]].astype('int'),sampling[features[i]].astype('int'),len_index[i-4])
            distribution_wasserstein.append(kl_div)
            distribution_mmd_01.append(kl_div)
            distribution_mmd_05.append(kl_div)
            distribution_mmd_10.append(kl_div)
            distribution_mmd_50.append(kl_div)
    
    result_summary=dict(zip(features,distribution_wasserstein))
    #result_summary['privacy_loss']=privacy_loss(train_dataset[:,sorted_index],sampling.to_numpy(),test_dataset[:,sorted_index])

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
    result_summary['sample_time']=str(t2-t1)
    with open(save_file+"result_sum.json", 'w') as file:
        json.dump(result_summary, file, indent=4)