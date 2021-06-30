# Unpaired MR Motion Artifact Deep Learning Using Outlier-Rejecting Bootstrap Aggregation

This repository is the official tensorflow implementation of "Unpaired MR Motion Artifact Deep Learning Using Outlier-Rejecting Bootstrap Aggregation".

> "Unpaired MR Motion Artifact Deep Learning Using Outlier-Rejecting Bootstrap Aggregation", 
> Gyutaek Oh, Jeong Eun Lee, and Jong Chul Ye, IEEE TMI [[Paper]](https://ieeexplore.ieee.org/abstract/document/9456930)

## Requirements

The code is implented in Python 3.5 with below packages.
```
tensorflow-gpu      1.13.1
numpy               1.16.4
scipy               1.1.0
tqdm                4.48.2
```

## Datasets
Below datasets were used for our experiments.
- Human Connectome Project (HCP) (brain, magnitude only images):
https://db.humanconnectome.org/
- FastMRI (knee, complex images):
https://fastmri.org/
- Chungnam National Hospital (liver, magnitude only images)

Examples of HCP and FastMRI data for the inference exist in the directory 'Data/Magnitude/test' and 'Data/Complex/test'.

## Pre-trained Models
You can download the pre-trained models from the [link](https://drive.google.com/drive/folders/1S4P6luYkipBQXHy4Vgbmpy-kSpFP2Mxv?usp=sharing)
Copy downloaded models to the directory 'Results/'.

## Traning and Evaluation
To evaluate our pre-trained model, run the below commands.
```
sh Scripts/MR_motion_magnitude.sh
sh Scripts/MR_motion_complex.sh
```

To train the model, add the '--training' option in the script files.

