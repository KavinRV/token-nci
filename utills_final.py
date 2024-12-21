import torch
from torch.utils.data import Dataset
import tqdm
import numpy as np
import argparse
import pandas as pd
import pickle
import torch
import random


def suffix_encoder(tokenizer, text, max_length, batching = False, prev_space = True):
    encoded = tokenizer(text, padding="max_length", max_length=88, truncation=True, return_tensors='pt')
    return encoded



def suffix_decoder(tokenizer, encoded):
    text = tokenizer.decode(encoded, skip_special_tokens=True)
    return text


def prefix_encoder(tokenizer, text, max_length=64, batch = False):
    encoded = tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors='pt')
    return encoded

def merge_prefix_suffix(prefix, suffix):
    if(len(suffix)>0 and len(prefix)>0 and suffix[0] == " " and prefix[-1] == " "):
        return prefix[:-1] + suffix
    else:
        return prefix + suffix



class Node(object):
    def __init__(self, token_id) -> None:
        self.token_id = token_id
        self.children = {}

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.token_id) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return '<tree node representation>'
    
class TreeBuilder(object):
    def __init__(self) -> None:
        self.root = Node(0)

    def build(self) -> Node:
        return self.root

    def add(self, seq) -> None:
        cur = self.root
        for tok in seq:
            if tok == 0:  # reach pad_token
                return
            # print(seq)
            if tok not in cur.children:
                cur.children[tok] = Node(tok)
            cur = cur.children[tok]

class BrandBuilder(object):
    def __init__(self) -> None:
        self.root = Node(0)

    def build(self) -> Node:
        return self.root

    def add(self, seq) -> None:
        cur = self.root
        for tok in seq:
            if tok == 32100:  # reach " <|SEP|> "
                if tok not in cur.children:
                    cur.children[tok] = Node(tok)
                return
            # print(seq)
            if tok not in cur.children:
                cur.children[tok] = Node(tok)
            cur = cur.children[tok]


class AutocompleteDataset(Dataset):
    def __init__(self,
        tokenizer,
        split="train", 
        tkmax_length=512,
        infer=False,
        pred_type="kmeans",
        ):
        self.tokenizer = tokenizer
        self.pred_type = pred_type
        # preprocessing

        if split == "train":
            data_path = "Data_process/NQ_dataset/nq_train_doc_newid.tsv"
            doc_aug = "Data_process/NQ_dataset/NQ_doc_aug.tsv"
            abs_aug = "Data_process/NQ_dataset/nq_title_abs.tsv"
            qg_aug = "Data_process/NQ_dataset/NQ_512_qg.tsv"

            df = pd.read_csv(
                data_path,
                names=["query", "queryid", "oldid", "bert_kmeans"],
                encoding='utf-8', header=None, sep='\t',
                dtype={'query': str, 'queryid': str, 'oldid': str, "bert_kmeans": str}
                )
            
            df_doc = pd.read_csv(
                doc_aug,
                names=["query", "queryid", "oldid", "bert_kmeans"],
                encoding='utf-8', header=None, sep='\t',
                dtype={'query': str, 'queryid': str, 'oldid': str, "bert_kmeans": str}
                )
            
            df_abs = pd.read_csv(
                abs_aug,
                names=["query", "queryid", "oldid", "bert_kmeans"],
                encoding='utf-8', header=None, sep='\t',
                dtype={'query': str, 'queryid': str, 'oldid': str, "bert_kmeans": str}
                )
            
            df_qg = pd.read_csv(
                qg_aug,
                names=["query", "queryid", "oldid", "bert_kmeans"],
                encoding='utf-8', header=None, sep='\t',
                dtype={'query': str, 'queryid': str, 'oldid': str, "bert_kmeans": str}
                )
            
            # data_full = pd.concat([df, df_doc, df_abs, df_qg], ignore_index=True)
            data_full = df

            data_full = data_full.dropna()
            data_full = data_full.drop(columns=['queryid'])
            data_full = data_full.drop_duplicates()
            # shuffle the data
            data_full = data_full.sample(frac=1).reset_index(drop=True)
        
        if split == "val":
            data_path = "Data_process/NQ_dataset/nq_dev_doc_newid.tsv"
            abs_aug = "Data_process/NQ_dataset/nq_title_abs.tsv"
            
            data_full = pd.read_csv(
                data_path,
                names=["query", "queryid", "oldid", "bert_kmeans"],
                encoding='utf-8', header=None, sep='\t',
                dtype={'query': str, 'queryid': str, 'oldid': str, "bert_kmeans": str}
                )
            
            df_abs = pd.read_csv(
                abs_aug,
                names=["query", "queryid", "oldid", "bert_kmeans"],
                encoding='utf-8', header=None, sep='\t',
                dtype={'query': str, 'queryid': str, 'oldid': str, "bert_kmeans": str}
                )

        num_lab = 30
        seq_len = 6

        self.data = data_full
        self.infer = infer
        self.d_max = 0
        self.pred_type = pred_type
        
        self.doc_code_new = {}
        self.v_lst = {}
        self.doc_full = {}
        self.max_val = 0

        uniq_bert = list(df_abs[f"bert_{self.pred_type}"].unique())

        gst = ""
        if infer:
            gst = "infer"

        if True:
            for k in uniq_bert:

                new_k = '-'.join([c for c in str(k).split('-')])
                v = torch.tensor([int(n) for n in new_k.split("-")])

                if str(v.tolist()) not in self.v_lst:
                    self.v_lst[str(v.tolist())] = 1
                else:
                    self.v_lst[str(v.tolist())] += 1
                    print("Duplicate code found")

                v = v.tolist()
                new_v = []
                dst = 0
                for val in v:
                    new_val = num_lab*dst + val + 2
                    dst += 1
                    new_v.append(new_val)
                    if new_val > self.d_max:
                        self.d_max = new_val
                        # if new_val > 51:
                        #     print(dst, val, num_lab)
                    
                    if val > self.max_val:
                        self.max_val = val

                if self.v_lst[str(v)] > 1:
                    end_val = num_lab*dst + 2 + self.v_lst[str(v)] - 2
                    if end_val > self.d_max:
                        self.d_max = end_val
                    new_v = new_v + [end_val, 1]
                else:
                    new_v = new_v + [tokenizer.eos_token_id]
                    new_v += [tokenizer.pad_token_id]*(seq_len - len(new_v))
                    # new_v = v.tolist() + [v_lst[str(v.tolist())]]
                
                
                self.doc_code_new[new_k] = torch.tensor(new_v)

            self.tree = TreeBuilder()
            
            for v in self.doc_code_new.values():
                self.tree.add(v.tolist())
            
            self.root = self.tree.build()
            # save the tree
            print("Saving the tree")
            with open(f"{split}_{self.pred_type}_tree.pkl", "wb") as f:
                pickle.dump(self.tree, f)

            print("Saving the doc_code")
            
            self.code_doc = {str(v.tolist()): k for k, v in self.doc_code_new.items()}
            with open(f"{split}_{self.pred_type}_list_id.pkl", "wb") as f:
                pickle.dump(self.code_doc, f)
            
            print("Tree and doc_code saved")

            self.data["code"] = self.data[f"bert_{self.pred_type}"].apply(lambda x: self.doc_code_new[x] if x != " " else torch.tensor([tokenizer.pad_token_id]*seq_len))

            # sample the data first 100
            # self.data = self.data.sample(500).reset_index(drop=True)

        self.max_length = tkmax_length
    
    def get_children(self, input_ids, trie_dict):

        if len(input_ids) == 0:
            output = list(trie_dict.children.keys())
            if len(output) == 0:
                return [0]
            return output
        elif input_ids[0] in trie_dict.children:
            return self.get_children(input_ids[1:], trie_dict.children[input_ids[0]])
        else:
            return [0] 
    
    def sub_get(self, input_ids):
        input_ids_new = input_ids[1:]
        out =  self.get_children(input_ids_new, self.root)
        return out
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        curr_eg = self.data.iloc[idx]

        if self.infer:

            input_text = curr_eg["query"]
            input_encoded = prefix_encoder(self.tokenizer, input_text, max_length=24)
            
            return input_encoded, curr_eg["code"], curr_eg["queryid"]
        
        input_text = curr_eg["query"]
        input_encoded = prefix_encoder(self.tokenizer, input_text, max_length=32)
        
        target_encoded = curr_eg["code"]

        return input_encoded, target_encoded