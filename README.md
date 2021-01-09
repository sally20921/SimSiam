# A Simple Framework For Contrastive Learning of Visual Representations
## Dataset
## Model
## Dependency
- I use python3 (3.5.2) and python2 is not supported. 
- I use PyTorch (1.1.0), though tensorflow-gpu is necessary to launch tensorboard.
## Install
```
git clone --recurse-submodules (this repo)
cd $REPO_NAME/code
(use python >= 3.5)
pip install -r requirements.txt
```
## Data Folder Structure
```
code/
 cli.py
 train.py
 evaluate.py
 infer.py
 ...
data/
```
## How To Use
### Training
```
cd code
python cli.py train
````

### Evaluation
```
cd code
python cli.py evaluate --ckpt_name=$CKPT_NAME
````
- Substitute CKPT_NAME to your preferred checkpoint file, e.g., `ckpt_name=model_name_simclr_ckpt_3/loss_0.4818_epoch_15`
## Results

## Contact Me
To contact me, send an email to sally20921@snu.ac.kr
