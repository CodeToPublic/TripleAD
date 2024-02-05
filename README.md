# TripleAD

This is the code for TripleAD.

## Dataset

1) Amazon dataset is a co-purchase network (cf. [Comga: Community-aware attributed graph anomaly detection, WSDM](https://github.com/XuexiongLuoMQ/ComGA)).

2) YelpChi dataset is a transaction network (cf. [Rethinking graph neural networks for anomaly detection, ICML](https://github.com/squareRoot3/Rethinking-Anomaly-Detection)). 

3) CiteSeer dataset is a citation network (cf. [Anomaly detection on attributed networks via contrastive self-supervised learning, TNNLS](https://github.com/GRAND-Lab/CoLA)).

4) ACM dataset is a citation network (cf. [Inductive anomaly detection on attributed networks, IJCAI](https://dl.acm.org/doi/abs/10.5555/3491440.3491619)).

5) Flickr is a typical social network (cf. [Deep anomaly detection on attributed networks, SDM](https://epubs.siam.org/doi/pdf/10.1137/1.9781611975673.67)).

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



