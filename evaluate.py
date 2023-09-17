import os.path

from run import Runner
import sys
from argparse import ArgumentParser

def evaluate(config_name, gpu_id, saved_suffix, dataset, corpus, conll_test_path):
    runner = Runner(config_name, gpu_id, dataset, corpus=corpus)
    model = runner.initialize_model(saved_suffix)

    _, _, examples_test = runner.data.get_tensor_examples()
    stored_info = runner.data.get_stored_info()

    # runner.evaluate(model, examples_dev, stored_info, 0, official=True, conll_path=runner.config['conll_eval_path'])  # Eval dev
    # print('=================================')
    runner.evaluate(model, examples_test, stored_info, 0, official=True, conll_path=conll_test_path)  # Eval test


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--config", default="train_mtl")
    parser.add_argument("--checkpoint")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--dataset", default="ontonotes", help="Select from ['ontonotes', 'ontogum', 'wikicoref']")
    parser.add_argument("--corpus", default="test")
    args = parser.parse_args()

    corpus = args.corpus
    if args.dataset == "ontogum":
        conll_path = f"./data/ontogum/{corpus}.gum.english.v4_gold_conll"
    elif args.dataset == "ontogum9":
        conll_path = f"./data/ontogum9/{corpus}.gum.english.v4_gold_conll"
    elif args.dataset == "gum9":
        conll_path = f"./data/gum9/{corpus}.gum.english.v4_gold_conll"
    elif args.dataset == "ontonotes":
        conll_path = f"./data/{corpus}.english.v4_gold_conll"
    elif args.dataset == "wikicoref":
        conll_path = f"./data/wikicoref/wikicoref.v4_gold_conll"
    elif args.dataset == 'ontogum_bio':
        conll_path = f"./data/ontogum_bio/{corpus}.gum.english.v4_gold_conll"
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    evaluate(args.config, args.gpu, args.checkpoint, args.dataset, corpus, conll_path)
