import numpy as np
import scipy
from scipy.spatial.distance import cdist



def drr_TS(X, Y, loop_size=50):
    # 分离连续和离散特征
    continuous_X = X[:, :-4]
    discrete_X = X[:, -4:]

    continuous_Y = Y[:, :-4]
    discrete_Y = Y[:, -4:]

    n_samples = continuous_X.shape[0]
    min_distances = np.full(n_samples, 1000.0)
    batch_size = max(1, n_samples // loop_size)  # 确保批次大小至少为1
    for i in range(0, n_samples, batch_size):
        end_index = min(i + batch_size, n_samples)

        # 分批处理连续特征
        batch_continuous_X = continuous_X[i:end_index]
        continuous_distances = cdist(batch_continuous_X, continuous_Y, 'euclidean')

        # 分批处理离散特征
        batch_discrete_X = discrete_X[i:end_index]
        discrete_distances = cdist(batch_discrete_X, discrete_Y, 'matching')
        #discrete_distances = (discrete_distances)

        # 合并连续和离散距离
        total_distances = np.sqrt(continuous_distances ** 2 + discrete_distances*4)
        #continuous_distances=0
        #discrete_distances=0
        #对每个点找到最小距离
        batch_min_distances = np.min(total_distances, axis=1)
        #total_distances=0
        min_distances[i:end_index] = batch_min_distances
    return min_distances

def privacy_loss(True_data,Sample_data,Test_data):
    memory=(drr_TS(True_data,Test_data)/(drr_TS(True_data,Sample_data)+1e-6))
    memory=np.mean(memory[memory>0])
    return memory

def calculate_kl_divergence(column1, column2, index_len):
    # 初始化概率分布数组
    prob_dist1 = np.zeros(index_len)
    prob_dist2 = np.zeros(index_len)

    # 计算每列的概率分布
    for value in column1:
        prob_dist1[value] += 1
    prob_dist1 /= len(column1)

    for value in column2:
        prob_dist2[value] += 1
    prob_dist2 /= len(column2)

    # 防止概率为0的情况，添加一个小的常数
    prob_dist1 += 1e-10
    prob_dist2 += 1e-10

    # 计算KL散度
    kl_div = scipy.stats.entropy(prob_dist1, prob_dist2)
    return kl_div