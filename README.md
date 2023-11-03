# Incorporating Singletons and Mention-based Features in Coreference Resolution via Multi-task Learning for Better Generalization

## Introduction
This repository contains the implementation of the paper from: 

- [Incorporating Singletons and Mention-based Features in Coreference Resolution via Multi-task Learning for Better Generalization](https://arxiv.org/pdf/2309.11582.pdf)
- [Yilun Zhu](http://yilunzhu.com/), [Siyao Peng](https://logan-siyao-peng.github.io/), [Sameer Pradhan](https://cemantix.org/), and [Amir Zeldes](https://corpling.uis.georgetown.edu/amir/)

## Prerequisites
1. Python >= 3.6
2. Install Python3 dependencies: pip install -r requirements.txt
3. Download the pretrained SpanBERT weights from [here](https://github.com/facebookresearch/SpanBERT) to `data/spanbert_large` .
4. Prepare datasets
    - Download & construct `v4_gold_conll` file
        - OntoNotes V5.0 from [here](https://catalog.ldc.upenn.edu/LDC2013T19). Run `setup_data.sh /path/to/ontonotes /path/to/data/dir`.
        - OntoGUM V8.1.0 from [here](https://github.com/amir-zeldes/gum/releases/tag/V8.1.0). Follow the instruction from [here](https://github.com/yilunzhu/ontogum) to construct the dataset. Within the OntoGUM/utils folder, run `python gum2conll.py` and the constructed file can be found in the `dataset` folder.
        - WikiCoref from [here](http://rali.iro.umontreal.ca/rali/?q=en/wikicoref). Go to `utils` under this repo and run `wikicoref2conll.py`.
    - Construct input files with various mention-level features for training
        - Go to `utils` and run `convert.py` for each dataset and type of mention information\
        For example, to acquire entity and information status for OntoGUM, run `python convert.py -i ./data/ontogum_sg -o ./data/ontogum_sg --discourse`\
    
    **please also ask the authors for pre-processed data.*

## Training
To run SpanBERT-large, you need at least 24GB GPU for training.
- `python run.py [config] [gpu_id]`
    - e.g., singleton + entity: `python run.py train_mtl_entity 0`
    - Models will be saved at data/[config]/model_[date-time].bin

## Evaluation
- `python evaluate.py [config] [model_id] [gpu_id]`
    - e.g. singleton + entity: `python evaluate.py train_mtl_entity Apr23_02-57-22_14000 0`

## Citation
```
@InProceedings{zhu-EtAl:2023:ijcnlp,
  author    = {Zhu, Yilun  and  Peng, Siyao  and  Pradhan, Sameer  and  Zeldes, Amir},
  title     = {Incorporating Singletons and Mention-based Features in Coreference Resolution via Multi-task Learning for Better Generalization},
  booktitle      = {Proceedings of the 13th International Joint Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics},
  month          = {November},
  year           = {2023},
  address        = {Nusa Dua, Bali},
  publisher      = {Association for Computational Linguistics},
  pages     = {121--130},
  url       = {https://aclanthology.org/2023.ijcnlp-short.14}
}
```
