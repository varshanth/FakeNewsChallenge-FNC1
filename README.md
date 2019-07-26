# Granularity-Based Prediction Framework with Stance Conditioned CNN for Fake News Classification - Stance Detection


<div align="center">
<img width="180" height="120" src="https://github.com/varshanth/FakeNewsChallenge-FNC1/raw/conditioned_cnn_experimental/figures/ourlogo_withoutsig.png" alt="Penrose Bros logo">
</div>

Authors/Contributors:  
* Varshanth R Rao  (Github: @varshanth)  
* Ritik Arora (Github: @ritikarora13)  
  
Stance detection describes the task of predicting the relative perspective of two pieces of text based on an issue or claim. Stance detection between the headline & body of a news article was the first stage in the Fake News Challenge issued in June 2017. Our novel granularity based prediction framework allows us to perform a 2 stage classification based on the granularity bucket. Apart from the FNC-1 baseline features, we extract & use additional intuitive textual features to assist our coarse stance Gradient Boosting Classifier. (Details given below). We also introduce a stance conditioned variant of the traditional Convolutional Neural Network (for NLP) to perform fine label classification. The stance conditioning assists the classifier to identify better separating hyperplanes by aligning the headline and body feature vectors. Visualization of the aligned vectors also helps us to understand why the model correctly/incorrectly classifies datapoints.
  

This repository is created based on the baseline implemenation.  
Link to the baseline implementation: [Baseline FNC implementation](https://github.com/FakeNewsChallenge/fnc-1-baseline)

Information about the fake news challenge could be found here: [FakeChallenge.org](http://fakenewschallenge.org)

## Getting Started
The FNC dataset is inlcuded as a submodule and can be FNC Dataset is included as a submodule. You should download the fnc-1 dataset by running the following commands. This places the fnc-1 dataset into the folder fnc-1/

    git submodule init
    git submodule update

## Usage
This project uses Python 3.6+ and PyTorch 1.0+.

### Main Dependencies
 ```
 pytorch 1.0
 numpy 1.13.1
 tqdm 4.15.0
 torchtext 0.3.1
 nltk 3.4
 scikit-learn 0.20.3
 bpemb 0.3.0
 scipy 1.2.1
 ```

Install dependencies:
```bash
pip install -r requirements.txt
```

### Run Pretrained Framework
```
python fnc_kfold.py -dl_weights_file benchmark_model_chkpts/cond_cnn_classif_disagree_light_3_BEST.pth -gb_weights_file model_chkpts/gb_weights_Q_pos_posSS_tfidf_tfidfSSMax_CCNN_bpe_2class.sav -run_2_class
```

#### Arguments
```
  -h, --help            show this help message and exit
  -c, --clean-cache     clean cache files
  -dl_weights_file DL_WEIGHTS_FILE
                        Path to DL Model Weights File
  -gb_weights_file GB_WEIGHTS_FILE
                        Path to GB Weights File
  -apply_pos_filter     Apply POS filters
  -run_2_class          Run the GB classifier for 2 labels - Related and
                        Unrelated
```

### Train or Test Only Stance Conditioned CNN
The model configuration can be found in dl_approach_cfg.py

```
python train_or_test_dl_model.py --help  
usage: train_or_test_dl_model.py [-h] [-test] [-weights_file WEIGHTS_FILE]  
                                 [-condition CONDITION] [-apply_pos_filter]  
  
CNN Based FNC Classifier  

optional arguments:  
  -h, --help            show this help message and exit  
  -test                 Activate Testing  
  -weights_file WEIGHTS_FILE  
                        Path to Weights File  
  -condition CONDITION  Label to Condition Network  
  -apply_pos_filter     Apply POS filters  
```

Condition can be one of {unrelated, agree, disagree, discuss}.  
If unrelated condition is chosen, the full dataset will be used for training.  
If other condition is used, all unrelated datapoints will NOT be considered.  
  
Example Usage for Training:
```
python train_or_test_dl_model.py -condition disagree  
```

Example Usage for Testing:  
```
python train_or_test_dl_model.py -test -condition disagree -weights_file benchmark_model_chkpts/cond_cnn_classif_disagree_light_3_BEST.pth  
```

## Model Architecture
<div align="center">
<img src="https://github.com/varshanth/FakeNewsChallenge-FNC1/blob/conditioned_cnn_experimental/figures/pipeline.png"><br><br>
</div>

## Handcrafted Features
<div align="center">
<img src="https://github.com/varshanth/FakeNewsChallenge-FNC1/blob/conditioned_cnn_experimental/figures/features.png"><br><br>
</div>

## Visualizations

* No Conditioning (λ= 0) Visualizations
<div align="center">
<img src="https://github.com/varshanth/FakeNewsChallenge-FNC1/blob/conditioned_cnn_experimental/figures/no_cond.png"><br><br>
</div>

* Light Conditioning (λ= 2) on “Discuss” Visualizations
<div align="center">
<img src="https://github.com/varshanth/FakeNewsChallenge-FNC1/blob/conditioned_cnn_experimental/figures/light_cond(lamda=2).png"><br><br>
</div>

* Aggressive Conditioning (λ= 100) on “Discuss” Visualizations
<div align="center">
<img src="https://github.com/varshanth/FakeNewsChallenge-FNC1/blob/conditioned_cnn_experimental/figures/aggr_cond(lambda=100).png"><br><br>
</div>

* Misclassified Data Points for Light Conditioning (λ= 2) on “Discuss” Visualizations
<div align="center">
<img src="https://github.com/varshanth/FakeNewsChallenge-FNC1/blob/conditioned_cnn_experimental/figures/discsuss_errors.png"><br><br>
</div>

## Results
<div align="center">
<img src="https://github.com/varshanth/FakeNewsChallenge-FNC1/blob/conditioned_cnn_experimental/figures/ablation_study.png"><br><br>
</div>
