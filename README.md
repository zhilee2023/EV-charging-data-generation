# EV-charging-data-generation
This repository contains the code for our research paper titled "Gibbs Sampling for Probabilistic Generation of Electric Vehicle Charging Data".

## Dependencies
The main dependencies are PyTorch (1.13.1+cu116) and GeoPandas (0.10.2).

## Dataset
Due to the sensitivity of the original dataset, we are unable to provide it here. However, we provide the synthetic dataset generated by our model. This data is intended to be used for training and is approximately equivalent to the original dataset in terms of size - roughly 1.6 million charging events across 3,777 battery electric vehicles over a period of 365 days. You can download the dataset from the following Google Drive link: https://drive.google.com/file/d/1K-ERurerpo-y02n9A3ygTOSDSfYTgeiB/view?usp=sharing


## Training
To train the model, first download the sample_best.csv file provided by us, and place it in the appropriate directory. Once downloaded, you can start the training process with the following command:

<pre>
python train.py
</pre>

## Data Generation
We provide a pretrained model that can be used for data generation. Two types of data generation are available: generating overall data and conditional generation.

For overall data generation, use the following command:
<pre>
python data_generation.py
</pre>

For conditional data generation, use the following command:
<pre>
python conditional_data_gen.py
</pre>

In conditional generation, specify the conditioning variables while running the script. More details can be found in the script's comments.

## Geo index
We also provide detailed boundary and information about 251 regions in Shanghai. You can visualize these regions using the provided script:

<pre>
import geopandas as gpd

gpd.read_file('shanghai_region/shanghai_boundary.shp',encoding='gbk')  
</pre>

We hope this repository helps to further research in the field of electric vehicle charging behavior. Please cite our paper if you find this code useful in your research.