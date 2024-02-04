# TripleAD

This is the code for the paper: “”.

## Dataset

1)Amazon dataset is a co-purchase network (cf. [Comga: Community-aware attributed graph anomaly detection, WSDM]()).

2)YelpChi dataset is a transaction network (cf. [Rethinking graph neural networks for anomaly detection, ICML]()). 

3)CiteSeer dataset is a citation network (cf. [Anomaly detection on attributed networks via contrastive self-supervised learning, TNNLS]()).

4)ACM dataset is a citation network (cf. [Inductive anomaly detection on attributed networks, IJCAI]()).

5)Flickr is a typical social network (cf. [Deep anomaly detection on attributed networks, SDM]()).

## Environment

Install Python 3.9.

```python
  pip install -r requirements.txt
```

## Run

Next, we demonstrate the results of the ACM dataset.

```python
  python main.py --dataset acm
```

By default, datasets are placed under the "dataset" folder. If you need to change the dataset, you can modify the dataset in the args.py file or use a command to specify the dataset. Here is an example command for changing the dataset:

```python
  python main.py --dataset datasetname
```

## RESULT

The GPU we used is NVIDIA RTX3090 24GB, and the following result is the AUC-ROC using the ACM dataset.

![](/fig/result.jpg "Title")


