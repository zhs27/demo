# Demo
This is the demo for my research on point cloud classification tasks.<br />



## Environment Set Up
The code is tested under linux system
Dependencies:<br />
Pytorch, tqdm, tensorboard, yaml, pytorch_wavelets, sklearn, opencv, pywt<br />
Create conda environment and activate<br />
```
conda create --name myenv
conda activate myenv
```
Install dependencies<br />
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tqdm
pip install tensorboard
pip install PyYAML
pip install pytorch-wavelets
pip install scikit-learn
pip install opencv-python
pip install PyWavelets
```
## Datasets
Download ScanObjectNN from this [link](https://drive.google.com/file/d/1xzh7a__wHvg6lUAWi-Hbanyt4XHPtw0Y/view)<br />
Download ModelNet40 from this [link](https://drive.google.com/file/d/10faoJ5rRT96Nhdqo9tGD3q7Vg_ZZ2apZ/view)<br />
Download ModelNet40 from this [link](https://drive.google.com/file/d/1EFbGbtmORogjbbQ22giChio3i_G5Oahk/view)<br />
Unzip the file after downloading<br />

## Arguments
Here are arguments you need to give:
```
--exp_name: the experiment name you give  
--dataset: you could fill one of ['ScanObjectNN','ModeNet40','ModeNet40C'].  
--data_path: the path of the dataset  
--k_way: the number of classes.  
--n_shot: the number of shots.  
```
# Run
Start running by:<br />
```
python main.py --exp_name <exp name> --dataset <dataset name> --data_path <your data path>  
```
If I missed any dependencies or if you have any questions please leave an issue to remind me.

