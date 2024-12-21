# import required libraries
import pandas as pd
import torch
from transformers import AutoTokenizer, T5Config
from transformers import T5ForConditionalGeneration as T5ForConditionalGenerationReal
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import argparse
import os
import wandb
from accelerate import Accelerator
import tqdm
import numpy as np
import namegenerator
import sys
import pickle
import random
from genret import T5ForConditionalGeneration
sys.path.append('./code')
from utills_final import AutocompleteDataset, merge_prefix_suffix, suffix_decoder
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

accelerator = Accelerator()
device = accelerator.device
print("PROCESS STARTED")

# accelerate launch --multi_gpu --num_processes 2 train_final.py  --model_dir t5-base/kmeans_1 --bs 128 --num_epochs 60 --model_name t5-base --pred_type kmeans --wandb
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --multi_gpu --num_processes 2 train_final.py  --model_dir t5-base_kmeans_1 --ckpt t5-base_kmeans_1/epoch_3.pth --bs 256 --num_epochs 60 --model_name t5-base --pred_type kmeans --wandb
# CUDA_VISIBLE_DEVICES=3,4 accelerate launch --multi_gpu --main_process_port 1320 --num_processes 2 train_final.py  --model_dir t5-base_kmeans_vq --vq --bs 256 --num_epochs 60 --model_name t5-base --pred_type kmeans --wandb
# CUDA_VISIBLE_DEVICES=2,7 accelerate launch --multi_gpu --main_process_port 1330 --num_processes 2 train_final.py  --model_dir t5-base_kmeans_oq_vq --vq --bs 256 --num_epochs 60 --model_name t5-base --pred_type kmeans --wandb


def get_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--bs',  type=int, default=4)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--tkmax_length', type=int, default=512)
    parser.add_argument('--mdmax_length', type=int, default=512)
    parser.add_argument('--initial_eval', action='store_true')
    parser.add_argument('--eval_every', type=int, default=300000000)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--model_name', type=str, default="t5-base")
    parser.add_argument('--pred_type', type=str, default="kmeans")
    parser.add_argument('--da', action='store_true')
    parser.add_argument('--vq', action='store_true')
    args = parser.parse_args()
    return args

def prepare_pred(pred, qid, dps):
    # pred B x Beams x L (list)
    # qid B x 1 (list)
    dct = {}
    for eg, id in zip(pred, qid):
        if id not in dct:
            dct[id] = []
        
        for beam in eg:
            #  make sure seq len is 6
            if len(beam) < 6:
                new_beam = beam + [0] * (6 - len(beam))
                print("bi")
            elif len(beam) > 6:
                new_beam = beam[:6]
                print("bi2")
            else:
                new_beam = beam

            dct[id].append(dps[str(new_beam)])
    return dct

def prepare_gt(gt, qid, dps):
    # gt B x L (list)
    # qid B x 1 (list)
    dct = {}
    for eg, id in zip(gt, qid):
        if id not in dct:
            dct[id] = []

        # make sure seq len is 6
        if len(eg) < 6:
            new_eg = eg + [0] * (6 - len(eg))
            print("hi")
        elif len(eg) > 6:
            new_eg = eg[:6]
            print("hi2")
        else:
            new_eg = eg

        dct[id].append(dps[str(new_eg)])
    
    return dct

def recal_at_k(pred_dct, gt_dct, k):
    # pred_dct: qid -> list of beams
    # gt_dct: qid -> list of beams
    total = 0
    correct = 0
    for qid in pred_dct:
        pred = pred_dct[qid][:k]
        gt = gt_dct[qid]
        for cand in pred:
            # print(f"candid: {cand}, gt: {gt}")
            if cand == gt[0]:
                correct += 1
                break
        total += 1
    return correct / total

def mrr(pred_dct, gt_dct):
    # pred_dct: qid -> list of beams
    # gt_dct: qid -> list of beams
    total = 0
    score = 0
    for qid in pred_dct:
        pred = pred_dct[qid]
        gt = gt_dct[qid]
        for i, cand in enumerate(pred):
            if cand in gt:
                score += 1/(i+1)
        total += 1
    return score / total
        
def main(args):
    #come with a new interseting name every time it is none
    if(args.model_dir is None and args.ckpt is None and accelerator.is_main_process):
        args.model_dir = args.model_name + "-" + args.pred_type + namegenerator.gen()
        if(args.dev):
            args.model_dir += "-dev"
        os.makedirs(args.model_dir)
        # save args to model directory along with model name
        with open(os.path.join(args.model_dir, "args.txt"), "w") as f:
            f.write(args.model_name + "\n")
            f.write(str(args))
        
        print("The model directory is created!")
        print("Model Directory: ", args.model_dir)
    elif(args.model_dir is not None and args.ckpt is None and  accelerator.is_main_process):
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
            # save args to model directory along with model name
            with open(os.path.join(args.model_dir, "args.txt"), "w") as f:
                f.write(args.model_name + "\n")
                f.write(str(args))
            
            print("The model directory is created!")
            print("Model Directory: ", args.model_dir)

    
    print("Using device:", device)

    # with open(args.train_data, "r") as f:
    #   data = f.read()
    # dataset = data.split("\n")
    # train_data = pd.DataFrame(dataset)
    # sentences = train_data.values.flatten().tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, tkmax_length=args.tkmax_length)

    seq_len = 6

    # if args.pred_type == "mod":
    #     seq_len = 6
    # elif args.pred_type == "br":
    #     seq_len = 14

    print("Tokenizing sentences...")
    dataset = AutocompleteDataset(tkmax_length=args.tkmax_length, tokenizer=tokenizer, pred_type=args.pred_type)
    print("total size of train dataset: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=args.bs, num_workers=4)

    # Load pre-trained model and tokenizer
    config = T5Config.from_pretrained(args.model_name)
    config.decoder_vocab_size = 152
    config.docid_max_len = seq_len
    config.custom = True
    if args.vq:
        config.vq = True
    # model = T5ForConditionalGeneration.from_pretrained(args.model_name, config=config)
    model = T5ForConditionalGeneration(config=config)
    pretrain_model = T5ForConditionalGenerationReal.from_pretrained(args.model_name)
    pretrain_params = dict(pretrain_model.named_parameters())
    for name, param in model.named_parameters():
        if name.startswith(("shared.", "encoder.")):
            with torch.no_grad():
                param.copy_(pretrain_params[name])
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # load ckpt if any
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        initial_epoch = ckpt["epoch"] + 1
        args.model_dir = os.path.dirname(args.ckpt)
        print("model loaded from checkpoint")
        print("model directory: ", args.model_dir)
    else:
        initial_epoch = 0
        print("pretrained model loaded -- no checkpoint found")

    if(args.wandb and accelerator.is_main_process):
        key = "cb9214905ae1b9737fed5614df1d085d1ddee3b2"
        wandb.login(key=key, relogin=True)
        wandb.init(project="nci-syn", name=args.model_dir)
    else:
        wandb.init(project="nci-syn", name=args.model_dir, mode="disabled")

    # log to wandb the model directory
    wandb.config.update(args)

    print("Tokenizing validation sentences...")
    val_dataset = AutocompleteDataset(tkmax_length=args.tkmax_length, tokenizer=tokenizer, pred_type=args.pred_type, split="val", infer=True)
    print("total size of validation dataset: ", len(val_dataset))
    # val_dataloader = DataLoader(val_dataset, batch_size=args.bs//10)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs)
        
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # print("after optimizer")
    # print(model.shared.weight.shape)
    

    print("STARTING TRAINING")
    model, optimizer, dataloader, val_dataloader = accelerator.prepare(model, optimizer, dataloader, val_dataloader)

    prev_acc = -100
    best_epoch = -100
        
    for epoch in tqdm.tqdm(range(initial_epoch, args.num_epochs)):
        # Training phase
        model.train()
        # print("after train load")
        # print(model.shared.weight.shape)
        total_train_loss = 0
        latest_val_loss = 0
        iterations = 0
        for batch in tqdm.tqdm(dataloader):
            inputs, targets = batch
            # Prepare data
            input_ids = inputs['input_ids'].squeeze(1)
            attention_mask = inputs['attention_mask'].squeeze(1)
            # labels = targets['input_ids'].squeeze(1)
            labels = targets.squeeze(1)
            labels[labels == tokenizer.pad_token_id] = -100
            optimizer.zero_grad()
            # if args.da:
            #     # outputs, da_loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, da_labels=domain)
            #     loss = outputs.loss
            #     total_loss = loss + 0.2*da_loss
            # else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss = loss
            
            # if iterations > 310:
                # print("-*_*-"*1000)
                # print("loss: ", total_loss)
            accelerator.backward(total_loss)
            optimizer.step()
            rand_prob = random.random()
            if rand_prob < 0.002:
                # Decode and print input text
                input_text = [tokenizer.decode(input_ids[0], skip_special_tokens=True)]
                print(f"Input Text (Epoch: {epoch}, Iteration {iterations}): {input_text}")
                
                # Generate model output and decode
                with torch.no_grad():
                    # see if model has attribute modules
                    if hasattr(model, "module"):
                        model_output = model.module.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=seq_len+1, min_length=seq_len-3, prefix_allowed_tokens_fn = lambda batch_id, input_ids: dataset.sub_get(input_ids.tolist()))
                    else:
                        model_output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=seq_len+1, min_length=seq_len-3, prefix_allowed_tokens_fn = lambda batch_id, input_ids: dataset.sub_get(input_ids.tolist()))
                    output_text = model_output[0].detach().clone().cpu().tolist()
                    if len(output_text[1:]) < seq_len:
                        output_text = output_text[1:] + [tokenizer.pad_token_id] * (seq_len - len(output_text[1:]))
                    else:
                        output_text = output_text[1:]
                    
                    pred_doc = dataset.code_doc[str(output_text)]
                    prediction = pred_doc

                    gt_enc = labels[0].detach().clone().cpu()
                    gt_enc[gt_enc == -100] = tokenizer.pad_token_id
                    gt_enc = gt_enc.tolist()

                    if sum(gt_enc) != 0:
                        gt_doc = dataset.code_doc[str(gt_enc)]
                        gt_text = gt_doc
                    else:
                        gt_text = " "
                    # gt_text = suffix_decoder(tokenizer, gt_enc)
                    
                    print(f"Model Output (Epoch: {epoch}, Iteration {iterations}): {prediction}, Ground Truth: {gt_text}")
                
                wandb.log({
                    "training loss": loss.item(),
                    })
            
            total_train_loss += loss.item()
            iterations = iterations + 1
            # Evaluation phase

        avg_train_loss = total_train_loss / len(dataloader)

        if True:
            model.eval()
            pred_lst = []
            qid_lst = []
            trg_lst = []
            print("STARTING VALIDATION")
            with torch.no_grad():
                for batch in tqdm.tqdm(val_dataloader):
                    inputs, targets, qid = batch
                    # Prepare data
                    input_ids = inputs['input_ids'].squeeze(1)
                    attention_mask = inputs['attention_mask'].squeeze(1)
                    labels = targets.squeeze(1)

                    if hasattr(model, "module"):
                        outputs = model.module.generate(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                num_beams=10,
                                                num_return_sequences=10,
                                                prefix_allowed_tokens_fn = lambda batch_id, input_ids: dataset.sub_get(input_ids.tolist())
                                                )
                    else:
                        outputs = model.generate(input_ids=input_ids, 
                                                attention_mask=attention_mask,
                                                num_beams=10,
                                                num_return_sequences=10,
                                                prefix_allowed_tokens_fn = lambda batch_id, input_ids: dataset.sub_get(input_ids.tolist())
                                                )
                    outputs = outputs[:, 1:]

                    if labels.shape[1] > outputs.shape[1]:
                        # extend outputs with pad tokens
                        pad = torch.full((outputs.shape[0], labels.shape[1] - outputs.shape[1]), tokenizer.pad_token_id, dtype=torch.long).to(device)
                        outputs = torch.cat((outputs, pad), dim=1)
                    elif labels.shape[1] < outputs.shape[1]:
                        pad = torch.full((labels.shape[0], outputs.shape[1] - labels.shape[1]), tokenizer.pad_token_id, dtype=torch.long).to(device)
                        labels = torch.cat((labels, pad), dim=1)

                    outputs = outputs.view(-1, 10, outputs.shape[1])

                    labels[labels == -100] = tokenizer.pad_token_id
                    
                    pred_lst += outputs.tolist()
                    qid_lst += list(qid)
                    trg_lst += labels.tolist()
                
                pred_dct = prepare_pred(pred_lst, qid_lst, dataset.code_doc)
                gt_dct = prepare_gt(trg_lst, qid_lst, dataset.code_doc)

                recal_dct = {}
                metric_dct = {}
                for k in [1, 5, 10, 50, 100]:
                    recal_dct[k] = recal_at_k(pred_dct, gt_dct, k)
                    metric_dct[f"eval/recall@{k}"] = recal_dct[k]

                mrr_score = mrr(pred_dct, gt_dct)
                metric_dct["eval/mrr"] = mrr_score
                metric_dct["train/loss"] = avg_train_loss
                metric_dct["eval/accuracy"] = recal_dct[1]
                metric_dct["train/epoch"] = epoch + 1

            print(f"Epoch: {epoch}, Training Loss: {loss.item()}, Recall@1: {recal_dct[1]}, Recall@5: {recal_dct[5]}, Recall@10: {recal_dct[10]}, Recall@50: {recal_dct[50]}, Recall@100: {recal_dct[100]}, MRR: {mrr_score}")
            # also log perplexity
            wandb.log(metric_dct)
            
            model.train()

        if hasattr(model, "module"):
            _model = accelerator.unwrap_model(model.module)
            # _optimizer = optimizer.module
        else:
            _model = model
            # _optimizer = optimizer

        print(f"Epoch: {epoch}, Training Loss: {avg_train_loss}")
        wandb.log({
            "avg_train_loss": avg_train_loss,
            })
        # Save model checkpoint at the end of each epoch only if is the main process
        if(accelerator.is_main_process) and prev_acc <= recal_dct[1]:
            prev_acc = recal_dct[1]
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': _model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss
            }
            checkpoint_path = os.path.join(args.model_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at '{checkpoint_path}'")

            if best_epoch != -100 and os.path.exists(os.path.join(args.model_dir, f'epoch_{best_epoch}.pth')):
                os.remove(os.path.join(args.model_dir, f'epoch_{best_epoch}.pth'))
                print(f"Checkpoint removed at '{os.path.join(args.model_dir, f'epoch_{best_epoch}.pth')}'")

            best_epoch = epoch

if __name__ == "__main__":
    main(get_args())