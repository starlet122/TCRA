# TCRA

This is the PyTorch implementation of TCRA for Knowledge Graph Completion task,, as described in our paper: “A Unified Joint Approach with Topological Context Learning and Rule Augmentation for Knowledge Graph Completion”.(ACL 2024)

## Dependencies

- [PyTorch](https://pytorch.org/) >= 1.9.1
- [DGL](https://www.dgl.ai/) >= 0.7.2 (for graph neural network implementation)
- [Hydra](https://hydra.cc/) >= 1.1.1 (for project configuration)

## Model Training

```shell
# enter the project directory
cd TCRA
# FB15k-237
conda activate gnn
python code/run.py
nohup python -u code/run.py &

# WN18RR
nohup python code/run.py dataset=WN18RR &
# Kinship
nohup python code/run.py dataset=Kinship &
```

## Datasets

Datasets used are contained in the folder `data/dataset`. 

mined_rules.txt: These rules are mined by RNNLogic. Format: [rule_head, rule_body_list]. For example, $r_1 \land r_2 \rightarrow r_3$ can be represented as $[r_3,r_1,r_2]$

## Acknowledgement

The project is built upon [SE-GNN](https://github.com/renli1024/SE-GNN)
