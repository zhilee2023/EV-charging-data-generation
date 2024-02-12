import numpy as np

class MMD_loss:
    def __init__(self):
        pass

    
    def gaussian_kernel(self, x, y, sigma=0.1):
        # Reshape x and y for broadcasting
        x=np.array(x)
        y=np.array(y)

        x = x.reshape(-1, 1)
        y = y.reshape(1, -1)

        # Broadcasting to compute pairwise differences
        diff_square = (x - y) ** 2
           # Clip values to avoid overflow
        threshold = -10  # You may adjust this threshold
        diff_square = np.clip(diff_square, a_min= threshold, a_max=None)
        # Calculate Gaussian kernel
        kernel = np.exp(-diff_square / (2 * sigma ** 2))
        return kernel

    def mmd_loss(self, x, y, sigma=0.05):
        x_kernel = self.gaussian_kernel(x, x, sigma)
        y_kernel = self.gaussian_kernel(y, y, sigma)
        xy_kernel = self.gaussian_kernel(x, y, sigma)
        mmd = np.mean(x_kernel) + np.mean(y_kernel) - 2 * np.mean(xy_kernel)
        return mmd

    def multi_trunk_mmd_loss(self,x,y,sigma):
        mmd_loss=0
        for i in range(5):
            indices_x = np.random.choice(x.shape[0], 10000, replace=False)
            indices_y = np.random.choice(y.shape[0], 10000, replace=False)
            mmd_loss+=self.mmd_loss(indices_x,indices_y,sigma)

        return mmd_loss/5
            