<h1 align="center">Data-Driven Design of Protein-Like Single-Chain Polymer Nanoparticles</h1>
<h4 align="center">Rahul Upadhya*, Matthew J. Tamasi*, Elena Di Mare, N. Sanjeeva Murthy, Adam J. Gormley</h4>

<p align="center"> [<b><a href="https://www.chemrxiv.org/engage/chemrxiv/article-details/631f37eabe03b23be6f3014d">Paper</a></b>]

![SCNP_img](https://user-images.githubusercontent.com/113135749/190904376-19bcdb67-3166-4081-b9de-624ca610d8f1.jpg)

## Contents

- [Overview](#overview)
- [Data](#data)
- [System Requirements](#system-requirements)
- [Model Training Demo](#model-training-demo)
- [Issues](https://github.com/GormleyLab/Data_Driven_Design_of_SCNPs/issues)
- [License](#license)

  
## Overview
This repository contains all of the code and data included in the article "Data-Driven Design of Protein-Like Single-Chain Polymer Nanoparticles" published on Chemrxiv Sep 13th, 2022.

## Data
The [data directory](https://github.com/GormleyLab/Data-Driven-Design-of-SCNPs/blob/main/Data/SCNP_DLS_SAXS_Data.csv) includes one .csv files that contain data regarding physical attributes of SCNPs captured using dynamic light scattering (DLS), small angle x-ray scattering (SAXS), and chemical descriptor calculations. This is the data utilized in the manuscript to train evidential regression models for SCNP property prediction.

## System Requirements
### Hardware Requirements
This code only requires a standard computer with enough RAM to support in-memory operations. However, as multiple machine learning pipelines occur here, we recommend running the code on systems with sufficient RAM and processing capacity:
  
RAM: 16+ GB
  
CPU: 4+ cores, 3.0+ GHz / core
  
Runtimes can be further improved through the usage of dedicated GPUs for model training.
  
## Software Requirements
### OS Requirements
The code was developed and tested only on *Windows 10*.
  
### Python Dependencies
```
numpy
scikit-learn
pandas
keras
```
## Model Training Demo
To train a model for SCNP prediction on the provided dataset, the following file can be run:

`SCNP_demo.py`

This will run X epochs of taining for a predefined evidential neural network and provide predictions on 20% of the original data held out. These results will be saved in a pickle file.
  
## References
The specific application, data, and machine learning models are described in: Data Driven Design of Single Chain Polymer Nanoparticles by ^Upadhya, R.; ^Tamasi, M.J.; Di Mare, E.; Murthy, N.S.; *Gormley, A.J (^ denotes equal contributions, * denotes corresponding authors), Chemrxiv, 2022.


## Help, suggestions, and corrections?
If you need help, have suggestions, identify issues, or have corrections, please send your comments to Prof. Adam Gormley at adam.gormley@rutgers.edu

## License
  
This project is covered under the **Apache 2.0 License**.
