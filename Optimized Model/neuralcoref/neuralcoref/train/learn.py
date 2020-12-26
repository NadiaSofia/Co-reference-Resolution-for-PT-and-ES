"""Conll training algorithm"""

import os
import time
import argparse
import socket
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from neuralcoref.train.model import Model
from neuralcoref.train.dataset import (
    NCDataset,
    NCBatchSampler,
    load_embeddings_from_file,
    padder_collate,
    SIZE_PAIR_IN,
    SIZE_SINGLE_IN,
)
from neuralcoref.train.utils import SIZE_EMBEDDING
from neuralcoref.train.evaluator import ConllEvaluator

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
STAGES = ["allpairs", "ranking"]


def get_all_pairs_loss(n):
    def all_pair_loss(scores, targets):
        """ All pairs and single mentions probabilistic loss
        """
        labels = targets[0]
        weights = targets[4].data if len(targets) == 5 else None
        loss_op = nn.BCEWithLogitsLoss(weight=weights, reduction="sum")
        loss = loss_op(scores, labels)

        return loss/n

    return all_pair_loss

def get_ranking_loss(n):
    def ranking_loss(scores, targets):
        """ Slack-rescaled max margin loss
        """
        costs = targets[1]
        true_ants = targets[2]
        weights = targets[4] if len(targets) == 5 else None
        true_ant_score = torch.gather(scores, 1, true_ants)
        top_true, _ = true_ant_score.max(dim=1)
        tmp_loss = scores.add(1).add(
            top_true.unsqueeze(1).neg()
        )  # 1 + scores - top_true
        if weights is not None:
            tmp_loss = tmp_loss.mul(weights)
        tmp_loss = tmp_loss.mul(costs)
        loss, _ = tmp_loss.max(dim=1)
        out_score = torch.sum(loss)
        return out_score / n

    return ranking_loss

def load_model(model, path):
    print("Reloading model from", path)
    model.load_state_dict(
        torch.load(path)
        if args.cuda
        else torch.load(path, map_location=lambda storage, loc: storage)
    )

def run_model(args):
    # Load datasets and embeddings
    embed_path = args.weights if args.weights is not None else args.train
    # N: Get embeddings + words from train words_vocab.txt
    tensor_embeddings, voc = load_embeddings_from_file(embed_path + "words")
    
    # N: Load datasets, using npy files
    dataset = NCDataset(args.train, args)
    eval_dataset = NCDataset(args.eval, args)
    test_dataset = NCDataset(args.test, args)

    print("Vocabulary:", len(voc))

    # Construct model
    print("Build model")
    model = Model(
        len(voc),
        SIZE_EMBEDDING, # N: 300
        args.h1, # N: 1000
        args.h2, # N: 500
        args.h3, # N: 500
        SIZE_PAIR_IN, # N: 2 * SIZE_MENTION_EMBEDDING + SIZE_FP
        SIZE_SINGLE_IN, # N: SIZE_MENTION_EMBEDDING + SIZE_FS
    )
    model.load_embeddings(tensor_embeddings)

    if args.cuda:
        model.cuda()

    if args.weights is not None:
        print("Loading pre-trained weights")
        model.load_weights(args.weights)

    if args.checkpoint_file is not None:
        print("Loading model from", args.checkpoint_file)
        model.load_state_dict(
            torch.load(args.checkpoint_file)
            if args.cuda
            else torch.load(
                args.checkpoint_file, map_location=lambda storage, loc: storage
            )
        )

    print("Loading conll evaluator")
    eval_evaluator = ConllEvaluator(
        model, eval_dataset, args.eval, args.evalkey, embed_path, args
    )
    train_evaluator = ConllEvaluator(
        model, dataset, args.train, args.trainkey, embed_path, args
    )
    test_evaluator = ConllEvaluator(
        model, test_dataset, args.test, args.testkey, embed_path, args
    )

    print("Testing evaluator and getting first train score")
    train_evaluator.test_model()

    print("Testing evaluator and getting first eval score")
    eval_evaluator.test_model()

    print("Testing evaluator and getting first test score")
    test_evaluator.test_model()

    # Preparing dataloader
    print("Preparing dataloader")
    print(
        "Dataloader parameters: batchsize",
        args.batchsize,
        "numworkers",
        args.numworkers,
    )
    batch_sampler = NCBatchSampler(
        dataset.mentions_pair_length, shuffle=True, batchsize=args.batchsize
    )
    dataloader = DataLoader(
        dataset,
        collate_fn=padder_collate,
        batch_sampler=batch_sampler,
        num_workers=args.numworkers,
        pin_memory=args.cuda,
    )
    mentions_idx, n_pairs = batch_sampler.get_batch_info()

    if args.test_model:
        model.eval()

        print("Model results of Eval Dataset\n")
        start_time = time.time()
        eval_evaluator.build_test_file()
        score, f1_conll, ident = eval_evaluator.get_score()
        elapsed = time.time() - start_time
        ep = elapsed / 60
        print(f"|| min/evaluation {ep:5.2f}")

        print("Model results of Test Dataset\n")
        start_time = time.time()
        test_evaluator.build_test_file()
        score, f1_conll, ident = test_evaluator.get_score()
        elapsed = time.time() - start_time
        ep = elapsed / 60
        print(f"|| min/evaluation {ep:5.2f}")

        return

    print("Start training!\n")
    improving = 1
    epoch = 0
    best_f1_conll = 0
    best_f1_all_pairs = 0
    best_f1_ranking = 0
    ending_all_pairs = 0
    ending_ranking = 0

    def run_epochs(loss_func, optim_func, save_name, lr, scheduler):
        nonlocal improving, epoch, best_f1_conll, best_f1_all_pairs, best_f1_ranking, ending_all_pairs, ending_ranking

        best_model_path = args.save_path + "best_model" + save_name
        final_model_path = args.save_path + "best_model"
        start_time_all = time.time()
        
        while improving:
            """ Run an epoch """
            print(f"{save_name} Epoch {epoch:d}")
            model.train()
            start_time_log = time.time()
            start_time_epoch = time.time()
            epoch_loss = 0

            for batch_i, (m_idx, n_pairs_l, batch) in enumerate(
                zip(mentions_idx, n_pairs, dataloader)
            ):
                inputs, targets = batch
                inputs = tuple(Variable(inp, requires_grad=False) for inp in inputs)
                targets = tuple(Variable(tar, requires_grad=False) for tar in targets)
                if args.cuda:
                    inputs = tuple(i.cuda() for i in inputs)
                    targets = tuple(t.cuda() for t in targets)
                scores = model(inputs)
                
                loss = loss_func(scores, targets)
                
                # Zero gradients, perform a backward pass, and update the weights.
                optim_func.zero_grad()
                loss.backward()
                epoch_loss += loss.item()
                optim_func.step()
                
                if batch_i % args.log_interval == 0 and batch_i > 0:
                    elapsed = time.time() - start_time_log
                    lr = optim_func.param_groups[0]["lr"]
                    ea = elapsed * 1000 / args.log_interval
                    li = loss.item()
                    print(
                        f"| epoch {epoch:3d} | {batch_i:5d}/{len(dataloader):5d} batches | "
                        f"lr {lr:.2e} | ms/batch {ea:5.2f} | "
                        f"loss {li:.2e}"
                    )
                    start_time_log = time.time()

            elapsed_epoch = time.time() - start_time_epoch
            ep = elapsed_epoch / 60
            print(
                f"|| min/epoch {ep:5.2f} | loss {epoch_loss:.2e}"
            )
            
            #Validate
            print("Evaluation Results!\n")
            model.eval()

            start_time = time.time()
            eval_evaluator.build_test_file()
            score, f1_conll, ident = eval_evaluator.get_score()
            elapsed = time.time() - start_time
            ep = elapsed / 60
            print(f"|| min/evaluation {ep:5.2f}")
                
            if ((f1_conll > best_f1_all_pairs and save_name == "allpairs") or (f1_conll > best_f1_ranking and save_name == "ranking")):
                print("New best model!")
                if save_name == "allpairs":
                    best_f1_all_pairs = f1_conll
                    if optim_func.param_groups[0]["lr"] < 1.00e-7:
                        ending_all_pairs = 0
                        print("Ending All-Pairs = 0\n")
                elif save_name == "ranking":
                    best_f1_ranking = f1_conll
                    if optim_func.param_groups[0]["lr"] < 1.00e-7:
                        ending_ranking = 0
                        print("Ending Ranking = 0\n")

                torch.save(model.state_dict(), best_model_path)
                if f1_conll > best_f1_conll:
                    best_f1_conll = f1_conll
                    torch.save(model.state_dict(), final_model_path)

            else:
                improving = 0
                if optim_func.param_groups[0]["lr"] < 1.00e-7:
                    if save_name == "allpairs":
                        ending_all_pairs += 1
                        print(f"Ending All-Pairs = {ending_all_pairs:3d}\n")
                    elif save_name == "ranking":
                        ending_ranking += 1
                        print(f"Ending Ranking = {ending_ranking:3d}\n")

            scheduler.step(f1_conll)
            epoch += 1

        improving = 1        
        load_model(model, best_model_path)


    if args.startstage is None or args.startstage == "allpairs":
        optimizer_a = Adam(model.parameters(), lr=args.all_pairs_lr, weight_decay=args.all_pairs_l2)
        loss_func_a = get_all_pairs_loss(batch_sampler.pairs_per_batch)
        scheduler_a = ReduceLROnPlateau(optimizer_a, mode='max', factor=0.1, patience=5, verbose=True)

        run_epochs(
            loss_func_a,
            optimizer_a,
            "allpairs",
            args.all_pairs_lr,
            scheduler_a,
        )

    if args.startstage is None or args.startstage in ["ranking", "allpairs"]:
        optimizer_r = Adam(model.parameters(), lr=args.ranking_lr, weight_decay=args.ranking_l2)
        loss_func_r = get_ranking_loss(batch_sampler.mentions_per_batch)
        scheduler_r = ReduceLROnPlateau(optimizer_r, mode='max', factor=0.1, patience=5, verbose=True)

        run_epochs(
            loss_func_r,
            optimizer_r,
            "ranking",
            args.ranking_lr,
            scheduler_r,
        )

    while (ending_all_pairs <= 5 or ending_ranking <= 5):
        run_epochs(
            loss_func_a,
            optimizer_a,
            "allpairs",
            args.all_pairs_lr,
            scheduler_a,
        )
        run_epochs(
            loss_func_r,
            optimizer_r,
            "ranking",
            args.ranking_lr,
            scheduler_r,
        )


    final_model_path = args.save_path + "best_model"
    load_model(model, final_model_path)
    model.eval()

    print("Best model results of Eval Dataset\n")
    start_time = time.time()
    eval_evaluator.build_test_file()
    score, f1_conll, ident = eval_evaluator.get_score()
    elapsed = time.time() - start_time
    ep = elapsed / 60
    print(f"|| min/evaluation {ep:5.2f}")

    print("Best model results of Test Dataset\n")
    start_time = time.time()
    test_evaluator.build_test_file()
    score, f1_conll, ident = test_evaluator.get_score()
    elapsed = time.time() - start_time
    ep = elapsed / 60
    print(f"|| min/evaluation {ep:5.2f}")

if __name__ == "__main__":
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(
        description="Training the neural coreference model"
    )
    parser.add_argument(
        "--train", type=str, default=DIR_PATH + "/data/", help="Path to the train dataset"
    )
    parser.add_argument(
        "--eval", type=str, default=DIR_PATH + "/data/", help="Path to the eval dataset"
    )
    parser.add_argument(
        "--test", type=str, default=DIR_PATH + "/data/", help="Path to the test dataset"
    )
    parser.add_argument(
        "--evalkey", type=str, help="Path to an optional key file for scoring"
    )
    parser.add_argument(
        "--weights", type=str, help="Path to pre-trained weights (if you only want to test the scoring for e.g.)"
    )
    parser.add_argument(
        "--batchsize", type=int, default=2000, help="Size of a batch in total number of pairs"
    )
    parser.add_argument(
        "--numworkers", type=int, default=8, help="Number of workers for loading batches"
    )
    parser.add_argument(
        "--startstage", type=str, help='Start from a specific stage ("allpairs", "ranking")'
    )
    parser.add_argument(
        "--checkpoint_file", type=str, help="Start from a previously saved checkpoint file"
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Print info every X mini-batches"
    )
    parser.add_argument(
        "--seed", type=int, default=1111, help="Random seed"
    )
    parser.add_argument(
        "--costfn", type=float, default=0.8, help="Cost of false new"
    )
    parser.add_argument(
        "--costfl", type=float, default=0.4, help="Cost of false link"
    )
    parser.add_argument(
        "--costwl", type=float, default=1.0, help="Cost of wrong link"
    )
    parser.add_argument(
        "--h1", type=int, default=1000, help="Number of hidden unit on layer 1"
    )
    parser.add_argument(
        "--h2", type=int, default=500, help="Number of hidden unit on layer 2"
    )
    parser.add_argument(
        "--h3", type=int, default=500, help="Number of hidden unit on layer 3"
    )
    parser.add_argument(
        "--all_pairs_lr", type=float, default=1e-3, help="All-Pairs training initial learning rate"
    )
    parser.add_argument(
        "--ranking_lr", type=float, default=1e-3, help="Ranking training initial learning rate"
    )
    parser.add_argument(
        "--all_pairs_l2", type=float, default=1e-6, help="All-Pairs training l2 regularization"
    )
    parser.add_argument(
        "--ranking_l2", type=float, default=1e-5, help="Ranking training l2 regularization"
    )
    parser.add_argument(
        "--patience", type=int, default=5, help="Patience (epochs) before considering evaluation has decreased"
    )
    parser.add_argument(
        "--min_lr", type=float, default=2e-8, help="Minimum learning rate"
    )
    parser.add_argument(
        "--lazy", type=int, default=1, choices=(0, 1), help="Use lazy loading (1, default) or not (0) while loading the npy files"
    )
    parser.add_argument(
        "--test_model", type=int, default=0, help="Test the model on Eval and Test sets (1) or not (0, default) - no training"
    )

    args = parser.parse_args()
    args.lazy = bool(args.lazy)
    args.costs = {"FN": args.costfn, "FL": args.costfl, "WL": args.costwl}
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    args.save_path = os.path.join(
        PACKAGE_DIRECTORY,
        "checkpoints",
        current_time + "_" + socket.gethostname() + "_",
    )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.evalkey = args.evalkey if args.evalkey is not None else args.eval + "/key.txt"
    args.trainkey = args.train + "/key.txt"
    args.testkey = args.test + "/key.txt"
    args.train = args.train + "/numpy/"
    args.eval = args.eval + "/numpy/"
    args.test = args.test + "/numpy/"
    run_model(args)
