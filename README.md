# Concise-CUT

## Original Paper
[Contrastive Learning for Unpaired Image-to-Image Translation](https://arxiv.org/pdf/2007.15651)

## Getting started

- Clone this repo:
```bash
git clone https://github.com/haomengz/Concise-CUT Concise-CUT
cd Concise-CUT
```

- Setup a virtual environment and install the requirements:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
echo $PWD > env/lib/python3.8/site-packages/Concise-CUT.pth
...
deactivate
```

## Train / Test

- Download a dataset:
```bash
bash ./datasets/download_cut_dataset.sh [DATASET_NAME]
```

- Train a model:
```bash
python train.py
```

- Test the model:
```bash
python test.py
```

## Different loss
Change the ```LOSS``` in config.py


## Acknowledgments
Code is inspired by [taesungp](https://github.com/taesungp/contrastive-unpaired-translation) and [wilbertcaine](https://github.com/wilbertcaine/CUT).

## Reference
```
@misc{park2020contrastive,
      title={Contrastive Learning for Unpaired Image-to-Image Translation}, 
      author={Taesung Park and Alexei A. Efros and Richard Zhang and Jun-Yan Zhu},
      year={2020},
      eprint={2007.15651},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```