"""Conll Evaluation - Scoring"""

import os
import subprocess
import io
import pickle

import torch
from torch.utils.data import DataLoader

from neuralcoref.train.conllparser import FEATURES_NAMES
from neuralcoref.train.dataset import NCBatchSampler, padder_collate
from neuralcoref.train.compat import unicode_

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

OUT_PATH = os.path.join(PACKAGE_DIRECTORY, "test_corefs.txt")
ALL_MENTIONS_PATH = os.path.join(PACKAGE_DIRECTORY, "test_mentions.txt")
SCORING_SCRIPT = os.path.join(PACKAGE_DIRECTORY, "scorer_wrapper.pl")

METRICS = ["muc", "bcub", "ceafm", "ceafe", "blanc"]
CONLL_METRICS = ["muc", "bcub", "ceafe"]

PT_DOCS = []

class ConllEvaluator(object):
    def __init__(self, model, dataset, test_data_path, test_key_file, embed_path, args):
        """ Evaluate the pytorch model that is currently being build
            We take the embedding vocabulary currently being trained
        """
        self.test_key_file = test_key_file
        self.cuda = args.cuda
        self.model = model
        batch_sampler = NCBatchSampler(
            dataset.mentions_pair_length, batchsize=args.batchsize, shuffle=False
        )
        self.dataloader = DataLoader(
            dataset,
            collate_fn=padder_collate,
            batch_sampler=batch_sampler,
            num_workers=args.numworkers,
            pin_memory=args.cuda,
        )
        self.mentions_idx, self.n_pairs = batch_sampler.get_batch_info()
        self.load_meta(test_data_path)

    def load_meta(self, test_data_path):
        # Load meta files
        datas = {}
        if not os.listdir(test_data_path):
            raise ValueError("Empty test_data_path")
        bin_files_found = False
        print("Reading ", end="")
        for file_name in os.listdir(test_data_path):
            if ".bin" not in file_name:
                continue
            bin_files_found = True
            print(file_name, end=", ")
            with open(test_data_path + file_name, "rb") as f:
                datas[file_name.split(".")[0]] = pickle.load(f)
        if not bin_files_found:
            raise ValueError(f"Can't find bin files in {test_data_path}")
        print("Done")
        self.m_loc = datas[FEATURES_NAMES[9]]
        self.tokens = datas[FEATURES_NAMES[10]]
        self.lookup = datas[FEATURES_NAMES[11]]
        self.docs = datas[FEATURES_NAMES[12]]
        self.flat_m_idx = list(
            (doc_i, m_i) for doc_i, l in enumerate(self.m_loc) for m_i in range(len(l))
        )

    ###########################
    #### CLUSTER FUNCTIONS ####
    ###########################

    def _prepare_clusters(self):
        """
        Clean up and prepare one cluster for each mention
        """
        self.mention_to_cluster = list(
            list(range(len(doc_mentions))) for doc_mentions in self.m_loc
        )
        self.clusters = list(
            dict((i, [i]) for i in doc_mentions)
            for doc_mentions in self.mention_to_cluster
        )

    def _merge_coreference_clusters(self, ant_flat_idx, mention_flat_idx):
        """
        Merge two clusters together
        """
        doc_idx, ant_idx = self.flat_m_idx[ant_flat_idx]
        doc_idx2, mention_idx = self.flat_m_idx[mention_flat_idx]
        assert doc_idx2 == doc_idx
        if (
            self.mention_to_cluster[doc_idx][ant_idx]
            == self.mention_to_cluster[doc_idx][mention_idx]
        ):
            return
        remove_id = self.mention_to_cluster[doc_idx][ant_idx]
        keep_id = self.mention_to_cluster[doc_idx][mention_idx]
        for idx in self.clusters[doc_idx][remove_id]:
            self.mention_to_cluster[doc_idx][idx] = keep_id
            self.clusters[doc_idx][keep_id].append(idx)
        del self.clusters[doc_idx][remove_id]

    def remove_singletons_clusters(self):
        for doc_idx in range(len(self.docs)):
            remove_id = []
            kept = False
            for key, mentions in self.clusters[doc_idx].items():
                if len(mentions) == 1:
                    remove_id.append(key)
                    self.mention_to_cluster[doc_idx][key] = None
                else:
                    kept = True
            if not kept:
                print("No coreference found")
            for rem in remove_id:
                del self.clusters[doc_idx][rem]

    ########################
    #### MAIN FUNCTIONS ####
    ########################

    def get_max_score(self, batch):
        inputs, mask = batch
        if self.cuda:
            inputs = tuple(i.cuda() for i in inputs)
            mask = mask.cuda()
        self.model.eval()
        with torch.no_grad():
            scores = self.model(inputs, concat_axis=1)
            scores.masked_fill_(mask, -float("Inf"))
            _, max_idx = scores.max(
                dim=1
            )  # We may want to weight the single score with coref.greedyness
        return scores.cpu().numpy(), max_idx.cpu().numpy()

    def test_model(self):
        print("Test evaluator / print all mentions")
        print(ALL_MENTIONS_PATH)
        self.build_test_file(out_path=ALL_MENTIONS_PATH, print_all_mentions=True)
        self.get_score(file_path=ALL_MENTIONS_PATH)

    def build_test_file(self, out_path=OUT_PATH, remove_singleton=False, print_all_mentions=False):
        """ Build a test file to supply to the coreference scoring perl script"""

        print("Building test file")
        self._prepare_clusters()
        self.dataloader.dataset.no_targets = True
        if not print_all_mentions:
            print("Build coreference clusters")
            for sample_batched, mentions_idx, n_pairs_l in zip(self.dataloader, self.mentions_idx, self.n_pairs):
                scores, max_i = self.get_max_score(sample_batched)
                for m_idx, ind, n_pairs in zip(mentions_idx, max_i, n_pairs_l):
                    if (ind < n_pairs):  # the single score is not the highest, we have a match !
                        prev_idx = m_idx - n_pairs + ind
                        self._merge_coreference_clusters(prev_idx, m_idx)
            if remove_singleton:
                self.remove_singletons_clusters()
        self.dataloader.dataset.no_targets = False

        print("Construct test file")
        out_str = ""
        for doc, d_tokens, d_lookup, d_m_loc, d_m_to_c in zip(
            self.docs, self.tokens, self.lookup, self.m_loc, self.mention_to_cluster
        ):
            out_str += (
                "#begin document (" + doc["name"] + "); part " + doc["part"] + "\n"
            )
            for utt_idx, (c_tokens, c_lookup) in enumerate(zip(d_tokens, d_lookup)):
                for i, (token, lookup) in enumerate(zip(c_tokens, c_lookup)):
                    out_coref = ""
                    for m_str, mention, mention_cluster in zip(
                        doc["mentions"], d_m_loc, d_m_to_c
                    ):
                        m_start, m_end, m_utt, m_idx, m_doc = mention
                        if mention_cluster is None:
                            pass
                        elif m_utt == utt_idx:
                            if m_start in lookup:
                                out_coref += "|" if out_coref else ""
                                out_coref += "(" + unicode_(mention_cluster)
                                if (m_end - 1) in lookup:
                                    out_coref += ")"
                                else:
                                    out_coref += ""
                            elif (m_end - 1) in lookup:
                                out_coref += "|" if out_coref else ""
                                out_coref += unicode_(mention_cluster) + ")"
                    out_line = (
                        doc["name"]
                        + " "
                        + doc["part"]
                        + " "
                        + unicode_(i)
                        + " "
                        + token
                        + " "
                    )
                    out_line += "-" if len(out_coref) == 0 else out_coref
                    out_str += out_line + "\n"
                out_str += "\n"
            out_str += "#end document\n"

        # Write test file
        print("Writing in", out_path)
        with io.open(out_path, "w", encoding="utf-8") as out_file:
            out_file.write(out_str)

    def get_score(self, file_path=OUT_PATH):
        """ Call the coreference scoring perl script on the created test file"""
        
        print("Computing score")
        score = {}
        ident = None
        for metric_name in CONLL_METRICS:
            print(self.test_key_file)
            print(file_path)
            try:
                scorer_out = subprocess.check_output(
                    [
                        "perl",
                        SCORING_SCRIPT,
                        metric_name,
                        self.test_key_file,
                        file_path,
                    ],
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                )
            except subprocess.CalledProcessError as err:
                print("Error during the scoring")
                print(err)
                print(err.output)
                raise
            value, ident = scorer_out.split("\n")[-2], scorer_out.split("\n")[-1]
            NR, DR, NP, DP = [float(x) for x in value.split(" ")]
            ident_NR, ident_DR, ident_NP, ident_DP = [
                float(x) for x in ident.split(" ")
            ]
            precision = NP / DP if DP else 0
            recall = NR / DR if DR else 0
            F1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0
            )
            ident_precision = ident_NP / ident_DP if ident_DP else 0
            ident_recall = ident_NR / ident_DR if ident_DR else 0
            ident_F1 = (
                2 * ident_precision * ident_recall / (ident_precision + ident_recall)
                if ident_precision + ident_recall > 0
                else 0
            )
            score[metric_name] = (precision, recall, F1)
            ident = (
                ident_precision,
                ident_recall,
                ident_F1,
                ident_NR,
                ident_DR,
                ident_NP,
                ident_DP,
            )
        F1_conll = sum([score[metric][2] for metric in CONLL_METRICS]) / len(
            CONLL_METRICS
        )
        print(
            "Mention identification recall",
            ident[1],
            "<= Detected mentions",
            ident[3],
            "True mentions",
            ident[4],
        )
        print("Scores", score)
        print("F1_conll", F1_conll)
        return score, F1_conll, ident
