## An Attentive-based Generative Model for Medical Image Synthesis
> [**An attentive‑based generative model for medical image synthesis**](https://arxiv.org/pdf/2306.01562.pdf)
> 
# ADC-CycleGAN


## Installation
```
pip install -r ADC-cycleGAN_requirements.txt
pip install -r cluster_requirements.txt
```
I recommend you use two environments to install. Maybe some packages will conflict when you install them into one environment.

## Cluster 

Run the “cluster.ipynb” to implement the cluster. Before that, you should install the jupyter on 
your computer and nb_conda as well. Please use your computer path instead of my path.

Then, you will get the cluster dataset under “cluster_path” root. 

## Train ADC-cycleGAN
```
python -u ADC-cycleGAN.py --G_rate=5 --lambdaG=10 --date='CBAM12RS' --t_cluster=4 --n_cluster=1 --loop_number=1 --out_dir=/home/jiayuan/ADC-cycleGAN/result --save_dir=/home/jiayuan/ADC-cycleGAN/result/model --dataset_path=/home/jiayuan/ADC-cycleGAN/dataset/cluster
```
For “--date” parameters, you can set any word you want, it is will control the save file name and
used to distinguish the different files.

For “--t_cluster”, set the total cluster. You needn’t change, because we use the number of 
clusters 4 for experiments. If you want to reproduce our ablation study, you can change it to one 
number of 2-5, but you should go back to step 2 and change some code to generate the number 
of clusters 2-3 and 5 datasets.

For “--n_cluster”, set the number of cluster. you should change it between 1-4 for each loop.

For “--loop_number”, set the number of loop. We have 5 times experiments for each method, 
so this parameter should be run from 1 to 5. 

Finally, you should run the code with the following parameters:
Loop_number | t_cluster | n_cluster
------------|-----------|-----------
1           | 4         | 1
1           | 4         | 2
1           | 4         | 3
1           | 4         | 4
2           | 4         | 1
2           | 4         | 2
2           | 4         | 3
2           | 4         | 4 
...         | ...       | ...
5           | 4         | 1
5           | 4         | 2
5           | 4         | 3
5           | 4         | 4

I strongly recommend you write a shell file to run automatically.

Totally you should run 20 times for “ADC-cycleGAN.py”. You will get 40 models because, for 
each training, you will get two directions model. Each training required 10 hours in GTX 1080 Ti 
GPU.

## Test
After you get the models, you should evaluate the model one by one. Please run "evaluate.ipynb" and change "dataset_path", "weight_path", and "save_path" to your path. All 
the results will save in “evaluate.txt”. Then you can use excel to analyze the results. 


## Citation:
If you use this code for your research, please cite our paper:
> @article{wang2023attentive,
> <br>  title={An attentive-based generative model for medical image synthesis},
> <br>  author={Wang, Jiayuan and Wu, QM Jonathan and Pourpanah, Farhad},
> <br>  journal={International Journal of Machine Learning and Cybernetics},
> <br>  pages={3897–3910},
> <br>  year={2023},
> <br>  publisher={Springer}
}

