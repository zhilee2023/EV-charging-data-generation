from vae_method_train import vae_train
from bayes_method_train import bayes_train
from gibbs_method_train import gibbs_train

layer=3
hidden_size=32
n_comp=4
z_dim=2
beta=1
beta_func=True
method='bayes'
all=[(16,3),(16,4),(16,5),(32,3),(32,4),(32,5)]
kk=5
if method=='bayes':
    layer=2
    hidden_size,n_comp=all[0]
    bayes_train(layer,hidden_size,n_comp,beta_func)

elif method=='vae':
    for layer in [2]:
        for hidden_size in [32]:
            for n_comp in [5]:
                if hidden_size==32 and n_comp==5:
                    for z_dim in [4]:
                        for beta in [1,2,3]:
                            vae_train(layer,hidden_size,n_comp,z_dim,beta,beta_func)
                else:
                    vae_train(layer,hidden_size,n_comp,2,1,beta_func)
   
elif method=='gibbs':
    for layer in [2]:
        for hidden_size in [16,32]:
            for n_comp in [5]:
                gibbs_train(layer,hidden_size,n_comp,beta_func)