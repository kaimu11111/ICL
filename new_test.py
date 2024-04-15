# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import pickle as pkl
import random
import torch
import math
import json
import string
import logging
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from collections import Counter, defaultdict
import sentence_transformers

from metaicl.data import MetaICLData
from metaicl.new_model import MetaICLModel
from gpt3 import GPT3Model

from utils.data import load_data

def main(logger, args):
    assert (args.dataset is not None and args.task is None) or (args.dataset is None and args.task is not None)

    if args.gpt.startswith("gpt3"):
        metaicl_model = GPT3Model(args.gpt[5:], args.api, logger)
        add_newlines = True
    else:
        add_newlines = not args.gpt.startswith("gpt2")
        
        task_counts = None
        if args.prefix_embed_file is not None:
            model_dir = Path(args.prefix_embed_file).parent.absolute()
            if os.path.exists(os.path.join(model_dir, 'task2token.json')):
                with open(os.path.join(model_dir, 'task2token.json')) as f:
                    task_counts = json.load(f)

        metaicl_model = MetaICLModel(args.gpt, logger, args.out_dir, 
            soft_prefix=args.use_soft_prefix or args.use_soft_postfix, 
            n_tokens=args.n_prefix_tokens, prefix_embed_file=args.prefix_embed_file,
            task_counts=task_counts)
        print("--------------------------------------------------------------------")
        for p in metaicl_model.model.parameters():
            print(p)
        metaicl_model.cuda()
        metaicl_model.eval()

    if "most_similar" in args.prior:
        embedding_model = sentence_transformers.SentenceTransformer(args.embedding_model)
        embedding_model.cuda()
        embedding_model.eval()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # setup hyperparams for data
    max_length_per_example = 256
    if args.use_demonstrations:
        max_length = min(max_length_per_example * args.k, args.max_length)
    else:
        max_length = max_length_per_example

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.test_batch_size, max_length, max_length_per_example))

    if args.use_soft_prefix or args.use_soft_postfix:
        metaicl_data = MetaICLData(logger, args.gpt, args.method,
            args.use_demonstrations, args.use_instruction, args.k, max_length, 
            max_length_per_example, add_newlines=add_newlines, 
            n_prefix_tokens=args.n_prefix_tokens, prefix=args.use_soft_prefix,
            task_counts=task_counts, prefix_token_ids=task_counts)
    else:
        metaicl_data = MetaICLData(logger, args.gpt, args.method,
            args.use_demonstrations, args.use_instruction, args.k,
            max_length, max_length_per_example, add_newlines=add_newlines)
    
    def load_configs():
        config_dict = {}
        for task in os.listdir("config/tasks"):
            if not task.startswith("unifiedqa:"):
                with open(os.path.join("config/tasks", task), "r") as f:
                    config = json.load(f)
                config_dict[task.split(".")[0]] = config
        return config_dict
    config_dict = load_configs()

    all_f1s = []
    all_accs = []
    errors = []
    all_scores = []
    all_dlls = []
    all_predictions = []
    seeds = args.seed.split(",")
    config_split = "unseen_domain_test" if args.unseen_domain_only else "test"

    for seed in seeds:
        # print("------------------------------------------beginning---------------------------------------------")
        np.random.seed(int(seed))

        ### data ...
        train_data = load_data(args.task, "train", args.k, 
            seed=seed, config_split=config_split,
            datasets=None if args.dataset is None else args.dataset.split(","),
            data_dir=args.data_dir, full_train=True)
        # print(f"Train data looks like: {train_data[:10]}")
        # print("-----------------------------------------------------------------")

        dev_data = load_data(args.task, args.split, args.k, seed=seed, config_split=config_split,
                             datasets=None if args.dataset is None else args.dataset.split(","), 
                             data_dir=args.data_dir, full_train=True)
        # print(f"dev data looks like: {dev_data[:10]}")
        # print("-----------------------------------------------------------------")

        if args.use_random_english_words:
            from english_words import get_english_words_set
            english_words_set = sorted(get_english_words_set(['web2']))

        train_counter = Counter()
        dev_counter = Counter()
        for dp in train_data:
            train_counter[dp["task"]] += 1
        for dp in dev_data:
            dev_counter[dp["task"]] += 1
        for k, v in train_counter.items():
            logger.info("[Train] %s\t%d" % (k, v))
        for k, v in dev_counter.items():
            logger.info("[Dev] %s\t%d" % (k, v))
        # print(f"Train counter looks like: {train_counter}")
        # print("-----------------------------------------------------------------")
        # print(f"dev counter looks like: {dev_counter}")
        # print("-----------------------------------------------------------------")

        logger.info("%s on %s (%d train, %d dev)" % (args.method, args.task, len(train_counter), len(dev_counter)))

        for test_task in dev_counter:
            curr_dev_data = [dp for dp in dev_data if dp["task"]==test_task]
            assert len(curr_dev_data)>0
            if args.test_size < len(curr_dev_data) and args.split=="test":
                subsample_ids = np.random.choice(len(curr_dev_data), args.test_size, replace=False)
                curr_dev_data = np.array(curr_dev_data)[subsample_ids].tolist()

            config_file = "config/tasks/{}.json".format(test_task)
            assert os.path.exists(config_file), config_file
            with open(config_file, "r") as f:
                config = json.load(f)
            is_classification = config["task_type"] == "classification"
            is_multi_choice = config["task_type"] == "multi-choice"
            if is_classification:
                options = curr_dev_data[0]["options"]
                assert np.all([d["options"]==options for d in curr_dev_data])

            if args.load_dir is None:
                if len(seeds) > 1:
                    save_path = os.path.join(args.out_dir, 
                        f"{test_task}-{metaicl_data.method}-{args.split}-s={seed}")
                else:
                    save_path = os.path.join(args.out_dir, 
                        f"{test_task}-{metaicl_data.method}-{args.split}")
            else:
                save_path = args.load_dir
            demonstrations = None
            # add by Rong  
            _train_data = [dp for dp in train_data if dp["task"]==test_task]
            subsample_ids = np.random.choice(len(_train_data), args.train_size, replace=False)
            curr_train_data = np.array(_train_data)[subsample_ids].tolist()
            demo_ids = np.random.choice(len(curr_train_data), 
                        args.k)

            demonstrations = []
            for i in demo_ids:
                demonstrations.append(curr_train_data[i])
                
            # add end
            f1, acc, pred, gt, nll, gt_label = run(test_task, metaicl_data, 
                        metaicl_model, demonstrations, curr_dev_data,
                        is_classification, save_path, config_dict)



            if save_path is not None and args.split=='train':
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                np.save(os.path.join(save_path, f'{args.split}-pred.npy'), pred)
                np.save(os.path.join(save_path, f'{args.split}-gt.npy'), gt_label)
                np.save(os.path.join(save_path, f'{args.split}-nll.npy'), nll)
                np.save(os.path.join(save_path, f'{args.split}-acc.npy'), acc)
#train_size
            all_predictions.append(pred)
            logger.info("%s task (seed=%s): Macro-F1: %.1f, Accuracy: %.1f" % 
                (args.task, seed, 100*f1, 100*acc))
            all_f1s.append(f1)
            all_accs.append(acc)
            # print(f"all_pridicitons looks like : {all_predictions[:10]} \n shape is {len(all_predictions)} * {len(all_predictions[0])}")
            # print("-----------------------------------------------------------------")
            # print("-------------------------------------end--------------------------------------------")
    final_predictions = []
    for p in np.transpose(all_predictions):
        v, c = np.unique(p, return_counts=True)
        # print(f"v is: \n {v}  \n c is:\n{c}")
        # print("-----------------------------------------------------------------")
        final_predictions.append(v[np.argmax(c)])
    final_f1, final_acc = metaicl_data.evaluate(final_predictions, gt, is_classification)
    logger.info("%s over %d target tasks with majority vote: Macro-F1: %.1f, Accuracy: %.1f" % 
        (args.task, len(all_f1s) // len(seeds), 100*final_f1, 100*final_acc))

    logger.info("%s over %d target tasks on average: Macro-F1: %.1f +- %.1f, Accuracy: %.1f +- %.1f" % 
        (args.task, len(all_f1s) // len(seeds), 100*np.mean(all_f1s), 100*np.std(all_f1s), 
        100*np.mean(all_accs), 100*np.std(all_accs)))
    


    if len(errors)>0:
        logger.info("You had errors with datasets:", ",".join(errors))
        logger.info("Please see the error messages")

def permutation(lst):
    if len(lst) == 0:
        return []
    if len(lst) == 1:
        return [lst]
 
    l = [] 
    for i in range(len(lst)):
       m = lst[i]
       remLst = lst[:i] + lst[i+1:]
       for p in permutation(remLst):
           l.append([m] + p)
    return l

def run(task, metaicl_data, metaicl_model, train_data, dev_data,
        is_classification, save_path, config_dict, return_all=False):

    if args.gpt.startswith("gpt3"):
        gpt3_dataloader, gpt3_metadata = metaicl_model.prepare_data(
            train_data if args.use_demonstrations else [],
            dev_data, args.method, batch_size=args.test_batch_size)
        losses, gpt3cache = metaicl_model.do_inference(gpt3_dataloader)	
        predictions, all_nlls, gt_labels, pred_labels = metaicl_model.do_predict(
            losses=losses, metadata=gpt3_metadata, return_nll=True)
    else:
        if args.use_instruction:
            instruction = config_dict[task]["instruction"]
            metaicl_data.tensorize(train_data, dev_data, instruction)
        else:
            metaicl_data.tensorize(train_data, dev_data)
        # metaicl_data.print_tensorized_example()
        losses = metaicl_model.do_inference(metaicl_data, args.test_batch_size)
        assert len(losses)==len(metaicl_data)
        predictions, all_nlls, gt_labels, pred_labels = metaicl_model.do_predict(
            metaicl_data, losses=losses, return_nll=True)
    try:
        groundtruths = [dp["output"] for dp in dev_data]
        f1, acc = metaicl_data.evaluate(predictions, groundtruths, 
            is_classification, return_all)
        return f1, acc, predictions, groundtruths, all_nlls, gt_labels
    except:
        return all_nlls, gt_labels

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_demonstrations", default=True, action="store_true")
    parser.add_argument("--use_instruction", default=False, action="store_true")
    parser.add_argument("--unseen_domain_only", default=False, action="store_true")
    parser.add_argument("--use_soft_prefix", default=True, action="store_true")
    parser.add_argument("--use_soft_postfix", default=False, action="store_true")
    parser.add_argument("--n_prefix_tokens", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--prior", type=str, nargs='+', default=[], 
        choices=["most_similar", "easiest", "hardest"])
    parser.add_argument("--difficulty", type=str, default="length", 
        choices=["concept_likelihood", "concept_calibrated"])
    parser.add_argument("--reorder", default=False, action="store_true")

    parser.add_argument("--log_dir", default='logs', type=str)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--load_dir", default=None, type=str)
    parser.add_argument("--concept_dir", default=None, type=str)
    parser.add_argument("--prefix_embed_file", default=None, type=str)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=str, default="100,13,21,42,87")

    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--global_step", type=str, default=None)
    parser.add_argument("--use_random_english_words", default=False, action="store_true")
    parser.add_argument("--use_random_label", default=False, action="store_true")

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--method", type=str, default="direct", 
        choices=["direct", "channel"])
    parser.add_argument("--gpt", type=str, default="gpt2-large", 
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
                "gpt3-ada", "gpt3-babbage", "gpt3-curie", "gpt3-davinci", 
                "gpt3-text-ada-001", "gpt3-text-babbage-001", "gpt3-text-curie-001", 
                "gpt3-text-davinci-001", "gpt3-text-davinci-002", 
                "gpt3-code-davinci-002", "gpt3-text-davinci-003"])
    parser.add_argument("--api", type=str, default=None)

    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--embedding_dir", type=str, default='embedding')
    parser.add_argument("--embedding_model", type=str, default='all-mpnet-base-v2', 
        choices=['all-mpnet-base-v2'])
    parser.add_argument("--similarity_temperature", type=float, default=1.0)
    parser.add_argument("--concept_temperature", type=float, default=10.0)
    # add
    parser.add_argument("--use_fixed_val", type=bool, default=True)
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, datetime.fromtimestamp(time.time()).isoformat())
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)