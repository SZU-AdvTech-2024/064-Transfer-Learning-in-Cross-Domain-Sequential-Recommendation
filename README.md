# TJAPL

The source code for our Information Sciences 2024 Paper [**"Transfer Learning in Cross-Domain Sequential Recommendation"**](https://www.sciencedirect.com/science/article/abs/pii/S0020025524004638).


## Environment

Our code is based on the following packages:
- GPU: Tesla V100-PCIe-32GB
- Requirments： 
   - Python = 3.8.13
   - PyTorch 1.7.1
   - pandas 1.3.4
   - numpy 1.21.3


## Usage

1. Download the datasets and put the files in `cross_data/amazon/`.

2. Run the data preprocessing scripts to generate the data. 
``` 
cd cross_data
python process.py 
```
More details on data processing can be found in `cross_data/README.md`.
3. To run the program, try the script given in 'train.sh'.
``` 
bash train.sh 
```
More descriptions of the command arguments are as follws:  
```
arg_name            | type      | description
--dataset             str         Name of the target domain (e.g. Books).  
--source_domain1      str         Name of the first source domain (e.g. Movies_and_TV). 
--source_domain2      str         Name of the second source domain (e.g. CDs_and_Vinyl).  
--num_epochs          int         Number of epochs.  
--batch_size          int         Batch size.  
--lr                  float       Learning rate.  
--device              str         Cpu or Cuda.  
--maxlen              int         Maximum length of sequences.  
--hidden_units        int         Latent vector dimensionality.  
--train_dir           str         Model to restore.  
--alpha               float       The weight of contrastive learning task.  
--beta                float       The weight of contrastive learning task. 
--gamma               float       The weight of contrastive learning task. 
--num_blocks          int         Number of attention blocks.  
--num_heads           int         Number of heads for attention.  
--dropout_rate        float       Dropout rate.  
--l2_emb              float       Regularization hyperparameter.  
```

## Cite

If you find this repo useful, please cite
```
@article{INS2024-TJAPL,
  title={Transfer Learning in Cross-Domain Sequential Recommendation},
  author={Zitao Xu and Weike Pan and Zhong Ming},
  journal={Information Sciences},
  volume={669},
  pages={120550},
  year={2024},
}
```