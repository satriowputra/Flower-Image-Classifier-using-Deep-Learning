# Image NN Classifier Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)

## Installation <a name="installation"></a>
Please provide workspace_utils.py if you run the code within Udacity workspace. To run code provided here, please install PyTorch on your system first. You should find no problem if you have Anaconda installed.

## Project Motivation<a name="motivation"></a>
Project code for Udacity's Introduction to Machine Learning with PyTorch Nanodegree program. In this project, I first develop code for an image classifier built with PyTorch, then convert it into a command line application.

By doing this project, i want to apply my skill and knowledge as Data Scientist to create image classifier based on neural network. 

## File Descriptions <a name="files"></a>
There are several file provided here which are:
1. `Image Classifier Project.ipynb`. This is where I create and try my code. There is code to train neural network classifier, save it, load it, and make prediction of a image. I also provided Image Classifier Project Outside.ipynb for you that run this code outside from Udacity workspace.
2. `cat_to_name.json` which contain labeled flower name.
3. `train.py` contains code to train neural network from selected data with details as follow:
> - Basic usage: `python train.py data_directory`
>> * Prints out training loss, validation loss, and validation accuracy as the network trains
> - Options:
>> - Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
>> - Choose architecture: `python train.py data_dir --arch "vgg13"`
>> - Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
>> - Use GPU for training: `python train.py data_dir --gpu`

4. `predict.py` predict name from selected image with details as follow:
> - Basic usage: `python predict.py /path/to/image checkpoint`
> - Options:
>> - Return top KK most likely classes: `python predict.py input checkpoint --top_k 3`
>> - Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
>> - Use GPU for inference: `python predict.py input checkpoint --gpu`

## Results<a name="results"></a>
If you want to display result with picture and and plots, please use the `Image Classifier Project.ipynb` file in your notebook.

If you want to run the in terminal / command line, please use `train.py` and `predict.py`. Output from the code will be list of predicted category and its probability.

Please take note that applicable CNN architecture is only `densenet161` or `vgg16` for now. Network accuracy should be around 80% with epoch higher than 3.

I also provided `Image Classifier Project Outside.ipynb` and `train_outside.py` for you who run the code outside from Udacity workspace. All other file should run fine on both workspace.