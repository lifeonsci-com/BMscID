
## BMscID: Bone marrow single cell ID annotation multi-classifier based on KNN label propagation semi-supervised machine learning algorithm

# Environment

1. at least python3.5+

# Input:

prepare three files below


1. metaData.txt: the label of the cells
2. train_data.txt： labeled gene expression profiles
3. unlabel_data.txt: unlabeled gene expression profiles

# Prepare

# prepare virtual env

```
python3 -m venv ./venv
source ./venv/bin/activate

pip install numpy 
pip install pandas 
pip install sklearn 
pip install imbalanced-learn
pip install matplotlib
```



# Run

```
git clone https://github.com/lifeonsci-com/BMscID.git

mkdir data
mkdir result

// move metaData.txt, train_data.txt and unlabel_data.txt under the data dir

// change the config of client_config.py 

python semi_classifier.py


```



















