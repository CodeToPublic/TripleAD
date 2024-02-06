# TripleAD

This is the code for TripleAD.

## Dataset

1) Amazon dataset is a co-purchase network from https://github.com/TienjinHuang/GraphAnomalyDetection/tree/master/data.

2) YelpChi dataset is a transaction network from https://github.com/zjunet/AMNet/tree/main/dataset. 

3) CiteSeer dataset is a citation network from https://github.com/GRAND-Lab/CoLA/tree/main/raw_dataset.

4) ACM dataset is a citation network from https://github.com/GRAND-Lab/CoLA/tree/main/dataset.

5) Flickr is a typical social network from https://github.com/GRAND-Lab/CoLA/tree/main/raw_dataset.

## Environment

The code is built in Python 3.9 and Pytorch 1.10.1. Please use the following command to install the requirements:

```python
  pip install -r requirements.txt
```

## Run

For the ACM dataset, the command to run is as follows:

```python
  python main.py --dataset acm
```

By default, datasets are placed under the "dataset" folder. If you need to change the dataset, you can modify the dataset in the args.py file or use a command to specify the dataset. Here is an example command for changing the dataset:

```python
  python main.py --dataset datasetname
```



