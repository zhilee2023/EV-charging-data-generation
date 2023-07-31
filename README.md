# EV-charging-data-generation
This repository contains the code for our research paper titled "Gibbs Sampling for Probabilistic Generation of Electric Vehicle Charging Data".

# Requirements
The codebase is implemented in Python. To install the necessary libraries, run the following command:

# Dependencies
The main dependencies are PyTorch and GeoPandas.

# Dataset
Due to the sensitivity of the original dataset, we are unable to provide it here. However, we provide the synthetic dataset generated by our model. This data is intended to be used for training and is approximately equivalent to the original dataset in terms of size - roughly 1.6 million charging events across 3,777 battery electric vehicles over a period of 365 days.

# Training
To train the model, use the following command:

<pre>
python train.py
</pre>

# Data Generation
The code supports both complete and conditional data generation. The complete data generation method provides synthetic data of similar scale to the original dataset. For conditional generation, specify the conditioning variables while running the script. More details can be found in the script's comments.

We hope this repository helps to further research in the field of electric vehicle charging behavior. Please cite our paper if you find this code useful in your research.
