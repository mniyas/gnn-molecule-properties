# Graph Neural Network for Predicting Molecular Properties
This project evaluates different Graph Neural Network(GNN) architectures for their effectiveness in predicting the Quantum Mechanical properties of chemical molecules.

## Requirements
We use PytorchLightning and PytorchGeometric as development frameworks and Weights & Biases for experiment management. 

## Setup
Install dependencies by running
 
```bash
grep -v '^#' requirements.txt | xargs -n 1 -L 1 pip install --default-timeout=100 --no-cache-dir
```

Set Python path using following command

```bash
export PYTHONPATH=.
```

If you are using Google Colab or Jupyter Notebook you can set the environment using

```bash
%env PYTHONPATH=.:$PYTHONPATH
```

## Run
To run DAGNN model use the following command

```bash
python training/experiment.py --model_class=MXMNet --data_class=PyG_QM9 --gpus='0,' --num_workers=4 --target=7 --lr=0.0001 --n_layer=2 --dagnn=True
```

To run the baseline model

```bash
python training/experiment.py --model_class=MXMNet --data_class=PyG_QM9 --gpus='0,' --num_workers=4 --target=7 --lr=0.0001 --n_layer=6
```

To run the model with Virtual Node

```bash
python training/experiment.py --model_class=MXMNet --data_class=PyG_QM9 --gpus='0,' --num_workers=4 --target=7 --lr=0.0001 --n_layer=6 --virtual_node=True
```

To run the model with Auxiliary Layer

```bash
python training/experiment.py --model_class=MXMNet --data_class=PyG_QM9 --gpus='0,' --num_workers=4 --target=7 --lr=0.0001 --n_layer=4 --auxiliary_layer=True
```
- If you are using Weights & Biases to track experiment, add `--wandb` flag as an argument.

- To perform distributed trainining with multiple gpus, add gpu card numbers like `--gpus=0,1,2,3` and accelerator as `--accelerator=ddp`


## References
- The base template of this codebase is taken from https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2021-labs
- MXMNet Implementation is taken from https://github.com/zetayue/MXMNet
- Auxiliary Layer implementation is inspired from https://github.com/rasbt/machine-learning-book/blob/main/ch18/ch18_part2.py
- Virtual Node and DAGNN Layer implementation is taken from https://github.com/divelab/MoleculeX/blob/master/BasicProp/kddcup2021/deeper_dagnn.py
