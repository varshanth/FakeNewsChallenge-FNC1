#<img src="https://github.com/varshanth/FakeNewsChallenge-FNC1/raw/conditioned_cnn_experimental/figures/ourlogo_withoutsig.png" alt="Penrose Bros logo" style="max-width:20%; float: left;
  margin: 5px 15px 0 0;"> Granularity-Based Prediction Framework with Stance Conditioned CNN for Fake News Classification - Stance Detection
Stance detection describes the task of predicting the relative perspective of two pieces of text based on an issue or claim. Stance detection between the headline \& body of a news article was the first stage in the Fake News Challenge issued in June 2017. In this paper, we elaborate on a novel granularity-based framework for stance detection. Apart from the FNC-1 Baseline features, we extract \& use additional intuitive textual features to assist our stance classifier. We also introduce a stance conditioned variant of the traditional Convolutional Neural Network used for NLP tasks in our framework.

Link to the baseline implementation: [Baseline FNC implementation](https://github.com/FakeNewsChallenge/fnc-1-baseline)

Some information about the fake news challenge could be found here: [FakeChallenge.org](http://fakenewschallenge.org)

## Getting Started
The FNC dataset is inlcuded as a submodule and can be FNC Dataset is included as a submodule. You should download the fnc-1 dataset by running the following commands. This places the fnc-1 dataset into the folder fnc-1/

    git submodule init
    git submodule update

## Usage
This project uses Python 3.5.3 and PyTorch 0.3.

### Main Dependencies
 ```
 pytorch 0.3
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

### Run
```
python fnc_kfold.py -dl_weights_file benchmark_model_chkpts/cond_cnn_classif_disagree_light_3_BEST.pth -gb_weights_file model_chkpts/gb_weights_Q_pos_posSS_tfidf_tfidfSSMax_CCNN_bpe_2class.sav -run_2_labels
```

### Arguments
```
  -h, --help            show this help message and exit
  -c, --clean-cache     clean cache files
  -dl_weights_file DL_WEIGHTS_FILE
                        Path to DL Model Weights File
  -gb_weights_file GB_WEIGHTS_FILE
                        Path to GB Weights File
  -apply_pos_filter     Apply POS filters
  -run_2_labels         Run the GDB classifier for 2 labels - Related and
                        Unrelated
```


## Model Architecture
<div align="center">
<img src="https://github.com/varshanth/FakeNewsChallenge-FNC1/blob/conditioned_cnn_experimental/figures/pipeline.png"><br><br>
</div>

## Handcrafted Features
<div align="center">
<img src="https://github.com/varshanth/FakeNewsChallenge-FNC1/blob/conditioned_cnn_experimental/figures/features.png"><br><br>
</div>

## Experiments

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