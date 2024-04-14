# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import matplotlib.pyplot as plt
import os
import argparse
import pickle as pkl
import random
import torch
import math
import json
import string
import logging
import numpy as np
import time
from datetime import datetime
import sentence_transformers
import torch.nn as nn
from sklearn import manifold,datasets
from collections import Counter, defaultdict
import pandas as pd

from transformers import GPT2Tokenizer, AutoTokenizer
from transformers import pipeline
from metaicl.data import MetaICLData
from metaicl.model import MetaICLModel
from utils.data import load_data

def main(logger, args):
    max_length_per_example = 256
    max_length = 256
    # if args.use_demonstrations:
    #     max_length = min(max_length * args.test_k, 1024)

    logger.info("batch_size=%d\tmax_length=%d\tmax_length_per_example=%d" % (
        args.batch_size, max_length, max_length_per_example))

    if args.dataset is not None:
        train_data = load_data(args.dataset, "train", args.k, seed=args.seed, 
                datasets=None if args.dataset is None else args.dataset.split(","),
                data_dir=args.data_dir)
    elif args.task is not None:
        train_data = load_data(args.task, "train", args.k, seed=args.seed, 
                config_split=args.split, data_dir=args.data_dir)
    else:
        print("please specify a train dataset/task.")
        exit(1)

    train_counter = Counter()
    for dp in train_data:
        train_counter[dp["task"]] += 1
    if args.local_rank <= 0:
        for k, v in train_counter.items():
            logger.info("[Train] %s\t%d" % (k, v))
        logger.info("%s on %s (%d train)" % (args.method, args.dataset, len(train_counter)))

    ######### load tensorize data
    metaicl_data = MetaICLData(logger, args.gpt2, args.method, args.use_demonstrations,
                               args.test_k, max_length, max_length_per_example,
                               do_tensorize=args.do_tensorize,
                               tensorize_dir=args.tensorize_dir,
                               n_process=args.n_process, n_gpu=args.n_gpu, 
                               local_rank=args.local_rank, 
                               n_prefix_tokens=args.n_prefix_tokens,
                               task_counts=train_counter)
    

        
    keyword = args.dataset if args.dataset is not None else args.task
    metaicl_data.tensorize_for_training(train_data, keyword=keyword, 
        seed=args.seed, use_random_english_words=args.use_random_english_words)

    if args.do_tensorize:
        return

    ######## actual training part

    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.train_seed)

    num_training_steps = args.num_training_steps
    save_period = 1000
    log_period = 100

    if args.no_masking:
        metaicl_data.tensorized_inputs["token_type_ids"] = torch.ones_like(metaicl_data.tensorized_inputs["input_ids"])
    # print("output is --------------", metaicl_data.print_tensorized_example(True))
    # shape = [[len(metaicl_data.print_tensorized_example(True)[i]), len(metaicl_data.print_tensorized_example(True)[i][0])]for i in metaicl_data.print_tensorized_example(True)]
    # print(shape)


    logger.info(args.out_dir)

    if args.local_rank<=0 and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    with open(os.path.join(args.out_dir, 'task2token.json'), 'w') as f:
        json.dump(metaicl_data.prefix_token_ids, f, ensure_ascii=False)

    metaicl_model = MetaICLModel(args.gpt2, logger, 
        args.out_dir, args.fp16, args.local_rank, True, args.n_prefix_tokens,
        prefix_embed_file=args.prefix_embed_file, task_counts=train_counter)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def embedding(input):
        '''input : [{'task': concept , 'input': sentence , 'output': 'ture/false', 'options': ['false', 'true']}]'''
        _tensorize_for_training = metaicl_data._tensorize_for_training
        out = _tensorize_for_training(input)
        embed_token = metaicl_model.model.get_input_embeddings()
        lm_logits = embed_token(torch.Tensor(out["input_ids"].to(device)).to(torch.int))
        '''
        output: (256: max_sentence_length, 1280: embedding length)
        example:
        tensor([[-0.0156, -0.0586, -0.0747,  ..., -0.0617,  0.0368,  0.0480],
         [ 0.0073,  0.0198, -0.0135,  ...,  0.0876,  0.0307,  0.0301],
         [-0.0339, -0.0170, -0.0435,  ...,  0.0215,  0.0891,  0.0368],
         ...,
         [-0.0149, -0.0209,  0.0021,  ...,  0.0336, -0.0005, -0.0090],
         [-0.0149, -0.0209,  0.0021,  ...,  0.0336, -0.0005, -0.0090],
         [-0.0149, -0.0209,  0.0021,  ...,  0.0336, -0.0005, -0.0090]],
       grad_fn=<EmbeddingBackward0>)
        '''
        return lm_logits[0]
    # task = [[{"task": "ag_news", "input": "Brazilian duo leave Celtic down and out DISASTER in Donetsk would be a catchy title, but it is unlikely that Celtic will be releasing a video of a Champions League match in which some initial bad luck was compounded by subsequent incompetence and disintegration, bringing the Parkhead side their", "output": "Sports", "options": ["Business", "Sci/Tech", "Sports", "World"]}],
    #         [{"task": "blimp-anaphor_number_agreement", "input": "Marla irritated herself.", "output": "acceptable", "options": ["acceptable", "unacceptable"]}],
    #         [{"task": "blimp-ellipsis_n_bar_2", "input": "Samuel isn't taking one mall and Kayla isn't taking several exposed mall.", "output": "acceptable", "options": ["acceptable", "unacceptable"]}],
    #         [{"task": "blimp-sentential_negation_npi_licensor_present", "input": "Claire did not ever resemble pictures.", "output": "acceptable", "options": ["acceptable", "unacceptable"]}],
    #         [{"task": "blimp-sentential_negation_npi_scope", "input": "Elizabeth's cousins that were worrying this bank should not ever want to clean some grocery store.", "output": "acceptable", "options": ["acceptable", "unacceptable"]}],
    #         [{"task": "ethos-directed_vs_generalized", "input": "Wish you cut your veins. Don\u2019t shout out you have mental problems. Act. Cut them", "output": "directed", "options": ["directed", "generalied"]}],
    #         [{"task": "ethos-disability", "input": "Wish you cut your veins. Don\u2019t shout out you have mental problems. Act. Cut them", "output": "true", "options": ["false", "true"]}],
    #         [{"task": "ethos-gender", "input": "Wish you cut your veins. Don\u2019t shout out you have mental problems. Act. Cut them", "output": "false", "options": ["false", "true"]}],
    #         [{"task": "ethos-race", "input": "Wish you cut your veins. Don\u2019t shout out you have mental problems. Act. Cut them", "output": "false", "options": ["false", "true"]}],
    #         [{"task": "ethos-religion", "input": "Wish you cut your veins. Don\u2019t shout out you have mental problems. Act. Cut them", "output": "false", "options": ["false", "true"]}],
    #         [{"task": "ethos-sexual_orientation", "input": "Wish you cut your veins. Don\u2019t shout out you have mental problems. Act. Cut them", "output": "false", "options": ["false", "true"]}],
    #         [{"task": "glue-cola", "input": "Fred knows which book for the children to read during the summer vacation.", "output": "unacceptable", "options": ["acceptable", "unacceptable"]}],
    #         [{"task": "glue-sst2", "input": "sentence: the cast ... keeps this pretty watchable , and casting mick jagger as director of the escort service was inspired .", "output": "positive", "options": ["negative", "positive"]}],
    #         [{"task": "poem_sentiment", "input": "less often man--the harder of the two.", "output": "no_impact", "options": ["negative", "no_impact", "positive"]}]
    #         ]
    # pd.set_option('display.max_columns', 1000) 
    # pd.set_option('display.width', 1000) 
    # pd.set_option('display.max_colwidth', 1000)
    # task_name = [i[0]["task"] for i in task]
    # result = np.zeros([14,14])
    # tensor_task = [torch.flatten(embedding(t)[:10]) for t in task]
    # for i in range(len(tensor_task)):
    #     for k in range(len(tensor_task)):
    #         sim = sentence_transformers.util.cos_sim(tensor_task[i],tensor_task[k])
    #         result[i][k] = sim
    
    # df_result = pd.DataFrame(result)
    # df_result.columns = task_name
    # df_result.index = task_name
    # df_result.to_csv('simlarity.csv', index=True)
    # print(df_result)
    # sentence_transformers.util.cos_sim(dev_embeddings, all_embeddings)
    # task_1 = [[{"task": "ethos-gender", "input": "Are you sure you are a girl? That wig on your legs say: I WANT TO BE A BOY", "output": "true", "options": ["false", "true"]}],
    #         [{"task": "ethos-gender", "input": "stone throwing must be legal when against women", "output": "true", "options": ["false", "true"]}],
    #         [{"task": "ethos-gender", "input": "All people are fucked up because of these queer thing because y'all realize how shitty gender roles are and how absurd is to see it in reverse genders roles", "output": "true", "options": ["false", "true"]}],
    #         [{"task": "ethos-gender", "input": "You should know women's sports are a joke", "output": "true", "options": ["false", "true"]}]]
    

    # task_2 = [[{"task": "blimp-sentential_negation_npi_scope", "input": "That convertible that can irritate Elaine has not ever stopped.", "output": "acceptable", "options": ["acceptable", "unacceptable"]}],
    #          [{"task": "blimp-sentential_negation_npi_scope", "input": "Melanie's lawyer who had taken every casserole might not ever arrive at that hospital.", "output": "acceptable", "options": ["acceptable", "unacceptable"]}],
    #          [{"task": "blimp-sentential_negation_npi_scope", "input": "This door that is alarming the guys has not ever opened.", "output": "acceptable", "options": ["acceptable", "unacceptable"]}],
    #          [{"task": "blimp-sentential_negation_npi_scope", "input": "A lot of cousins of Stacy who have driven to all hills have not ever cheered.", "output": "acceptable", "options": ["acceptable", "unacceptable"]}]]
    
    
    # task_3 = [[{"task": "ethos-race", "input": "After all this time the african americans are the people with most crime rate", "output": "true", "options": ["false", "true"]}]]
    # task_4 = [[{"task": "ethos-sexual_orientation", "input": "Hope you fucking die bitches gay people should not exist all you do is lick one another's ass", "output": "true", "options": ["false", "true"]}]]
    # task_5 = [[{"task": "glue-cola", "input": "This girl in the red coat or you will put a picture of Bill on your desk before tomorrow.", "output": "acceptable", "options": ["acceptable", "unacceptable"]}]]

    # task_6 = [[{"task": "ethos-religion", "input": "Why are we not allowed to kill christians?", "output": "true", "options": ["false", "true"]}],
    #           [{"task": "ethos-religion", "input": "Christians are all demonic", "output": "true", "options": ["false", "true"]}]]
    # task_7 = [[{"task": "ag_news", "input": "Rescuing an Old Saver If you think you may need to help your elderly relatives with their finances, don't be shy about having the money talk -- soon.", "output": "Business", "options": ["Business", "Sci/Tech", "Sports", "World"]}],
    #           [{"task": "ag_news", "input": "Chad seeks refugee aid from IMF Chad asks the IMF for a loan to pay for looking after more than 100,000 refugees from conflict-torn Darfur in western Sudan.", "output": "Business", "options": ["Business", "Sci/Tech", "Sports", "World"]}]]
    # tensor_task_1 = [torch.flatten(embedding(task)) for task in task_3]
    # tensor_task_2 = [torch.flatten(embedding(task)) for task in task_4]
    # tensor_task_3 = [torch.flatten(embedding(task)) for task in task_1]
    # sim = sentence_transformers.util.cos_sim(tensor_task_1[0],tensor_task_2[0])
    # sim1 = sentence_transformers.util.cos_sim(tensor_task_1[0],tensor_task_3[0])
    # sim2 = sentence_transformers.util.cos_sim(tensor_task_2[0],tensor_task_3[0])
    # print("similarity for ethos-race -- ethos-sexual_orientation is", sim)
    # print("similarity for ethos-race --  ethos-gender is", sim1)
    # print("similarity for ethos-sexual_orientation --  ethos-gender is", sim2)
    # tensor_task_1 = [embedding(task)[:10] for task in task_1]
    # tensor_task_2 = [embedding(task)[:10] for task in task_2]
    # tensor_task_3 = [embedding(task)[:10]  for task in task_3]
    # tensor_task_4 = [embedding(task)[:10]  for task in task_4]
    # tensor_task_5 = [embedding(task)[:10]  for task in task_5]
    # tensor_task_6 = [embedding(task)[:10] for task in task_6]
    # tensor_task_7 = [embedding(task)[:10]  for task in task_7]

    # tsne = manifold.TSNE(n_components=2,init="random",perplexity=20,metric="cosine")
    # X = torch.cat((tensor_task_1[0],tensor_task_2[0],tensor_task_3[0],
    #                tensor_task_4[0],tensor_task_5[0],tensor_task_6[0],tensor_task_7[0]), dim=0)
    # # # print(X)
    # result = tsne.fit_transform(X.cpu().detach().numpy())
    # # # print("shape is ", result.shape)
    # # # print(result)
    # x = result[:,0]
    # y = result[:,1]
    # plt.figure(figsize=(8, 5), dpi=80)
    # plt.scatter(x[:10],y[:10],c = "red",label="gender")
    # plt.scatter(x[10:20],y[10:20],c = "green",label = "sentential_negation_npi_scope")
    # plt.scatter(x[20:30],y[20:30],c = "black", label = "race")
    # plt.scatter(x[30:40],y[30:40],c = "orange",label = "sexual_orientation")
    # plt.scatter(x[40:50],y[40:50],c = "purple", label = "cola")
    # plt.scatter(x[50:60],y[50:60],c = "beige", label = "religion")
    # plt.scatter(x[60:70],y[60:70],c = "cyan", label ="news")
    # plt.legend(fontsize=8)
    
    # # # plt.xlabel('X-axis')
    # # # plt.ylabel('Y-axis')
    # plt.savefig('scatter_plot.png')
    # print("concept token :")
    # print(len(tensor_task_1[0]),len(tensor_task_1[0][0]))
    # print(tensor_task_2[0])
    # print(tensor_task_3[0])
    # mean_task_1 = (torch.sum(tensor_task_1[0],dim=0)) / 10
    # mean_task_2 = (torch.sum(tensor_task_2[0],dim=0)) / 10
    # mean_task_3 = (torch.sum(tensor_task_3[0],dim=0)) / 10
    # print("mean :")
    # # print(mean_task_1)
    # # print(mean_task_2)
    # sim1 = sentence_transformers.util.cos_sim(mean_task_1, mean_task_2)
    # sim2 = sentence_transformers.util.cos_sim(mean_task_1, mean_task_3)
    # sim3 = sentence_transformers.util.cos_sim(mean_task_2, mean_task_3)
    # pdist = nn.PairwiseDistance(p=2)
    # sim1 = pdist(mean_task_1,mean_task_2)
    # sim2 = pdist(mean_task_1,mean_task_3)
    # sim3 = pdist(mean_task_2,mean_task_3)
    # print("distance for ETHOS-R -- ETHOS-SO is ", sim1)
    # print("distance for ETHOS-R -- COLA is ", sim2)
    # print("distance for COLA -- ETHOS-SO is ", sim3)
    # print(tensor_task_1)
    # print(tensor_task_2)
    # sim1 = sentence_transformers.util.cos_sim(tensor_task_1[0], tensor_task_2[0])
    # sim2 = sentence_transformers.util.cos_sim(tensor_task_2[0], tensor_task_3[0])
    # sim3 = sentence_transformers.util.cos_sim(tensor_task_3[0], tensor_task_4[0])
    # # for i in range(len(tensor_task_1)-1):
    # #     for k in range(i+1,len(tensor_task_1)):
    # #         sim = sentence_transformers.util.cos_sim(tensor_task_1[i],tensor_task_1[k])
    # #         sim1 = sentence_transformers.util.cos_sim(tensor_task_2[i],tensor_task_2[k])
    # #         print(f"similarity for same task  is {sim}")
    # #         print(f"similarity for same task  is {sim1}")  
    # sim4 = sentence_transformers.util.cos_sim(tensor_task_4[0], tensor_task_5[0])
    # sim5 = sentence_transformers.util.cos_sim(tensor_task_5[0], tensor_task_6[0])
    # sim6 = sentence_transformers.util.cos_sim(tensor_task_6[0], tensor_task_7[0])
    
    # print("similarity ： ")
    # print(sim1)
    # print(sim2)  
    # print(sim3) 
    # print(sim4)
    # print(sim5)  
    # print(sim6)
    # print("similarity for different task")
    # for i in range(len(tensor_task_1)):
    #     for k in range(i,len(tensor_task_1)):
    #         sim = sentence_transformers.util.cos_sim(tensor_task_1[i],tensor_task_2[k])
    #         print("similarity for different task is", sim)
            
    # print(sim4)
    # print(sim5)
    # print(sim6)


    # pipe = pipeline('feature-extraction', model=metaicl_model.model, tokenizer=metaicl_data.tokenizer)
    # result = pipe("FUCK GAYS disgusting")
    # tensorized_example =[50257, 50258, 50259, 50260, 50261, 50262, 50263, 50264, 50265, 50266, 9562, 30998, 402, 4792, 50, 23374, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # print("token after embedding-------------",result)
    # print(f"the shape is {np.array(result).shape}")
    # embed_token = metaicl_model.model.get_input_embeddings()
    # lm_logits = embed_token(torch.Tensor(tensorized_example).to(torch.int))
    # print("-------------------",lm_logits)
    # print("-------shape------------",lm_logits.detach().numpy().shape)

    metaicl_model.to_device()
    metaicl_model.setup_optimizer(args.optimization, num_training_steps, args.lr,
                                  args.weight_decay, args.warmup_steps)
    metaicl_model.parallel()
    metaicl_model.train()

    metaicl_model.do_train(metaicl_data, args.batch_size, num_training_steps, save_period, log_period)
    metaicl_model.print()
    print('end')

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_tensorize", default=False, action="store_true")
    parser.add_argument("--tensorize_dir", type=str, default="tensorized")
    parser.add_argument("--n_gpu", type=int, default=8)
    parser.add_argument("--n_process", type=int, default=40)
    parser.add_argument("--n_prefix_tokens", type=int, default=10)

    parser.add_argument("--use_demonstrations", default=False, action="store_true")
    parser.add_argument("--log_dir", default='logs', type=str)
    parser.add_argument("--prefix_embed_file", default=None, type=str)

    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--test_k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--train_seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--no_masking", default=False, action="store_true")
    parser.add_argument("--use_random_english_words", default=False, action="store_true")

    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--method", type=str, default="direct", 
        choices=["direct", "channel", "causal", "anti-causal"])
    parser.add_argument("--gpt2", type=str, default="gpt2-large")

    parser.add_argument("--optimization", type=str, default="adamw")
    parser.add_argument("--fp16", default=False, action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    log_file = os.path.join(args.log_dir, datetime.fromtimestamp(time.time()).isoformat())
    handlers = [logging.StreamHandler(), logging.FileHandler(log_file)]

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=handlers)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    logger = logging.getLogger(__name__)
    logger.info(args)

    main(logger, args)
