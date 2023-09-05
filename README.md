# Metrics for Decision Maps

This repo contains the code to produce the results presented at the paper **Quantitative and Qualitative Comparison of Decision Map Techniques for Explaining Classification Models**.

Codes for DBM, SDBM and DeepView are included.

# Python Setup

- Install Anaconda Python
- Install tensorflow, and UMAP

```
python -m pip install -r requirements.txt
```

## Collecting the datasets


The real datasets are the MNIST, Fashion-MNIST, UCI HAR Dataset and the Reuters newswire classification dataset. Get the data by:

```
python get_data.py
```

## Running the experiments

Run the gloabal metrics evaluation by:

```
python run_metrics.py
```

## To do:
Part of the local metrics and visualzie the results
