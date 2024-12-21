from transformers import T5Tokenizer
import torch
import pandas as pd
from utills import AutocompleteDataset, suffix_decoder
from torch.utils.data import DataLoader
from utills import Node, TreeBuilder
from transformers import T5Tokenizer, T5Config
from genret import T5ForConditionalGeneration
import torch
import pandas as pd
from utills import AutocompleteDataset, suffix_decoder
import pickle
from tqdm import tqdm
import json
# from torch.utils.data import DataLoader

tokenizer = T5Tokenizer.from_pretrained('t5-large')
dataset_train = AutocompleteDataset(data_path="train.csv", tkmax_length=28, tokenizer=tokenizer, pred_type="mod")
print("total size of train dataset: ", len(dataset_train))
dataloader_train = DataLoader(dataset_train, batch_size=64, num_workers=4)

dataset_train_br = AutocompleteDataset(data_path="train.csv", tkmax_length=28, tokenizer=tokenizer, pred_type="br")
print("total size of train dataset: ", len(dataset_train))
dataloader_train_br = DataLoader(dataset_train, batch_size=64, num_workers=4)

epcs = [5, 6, 7, 8, 9]
beam = 2
cuda_select = 3 # select the cuda device
cp_mod = "t5-large-snazzy-magnolia-greyhound" # model for mod
cp_br = "t5-large-snazzy-magnolia-greyhound" # model for brand

for epc in epcs:
    config = T5Config.from_pretrained('t5-large')
    config.decoder_vocab_size = dataset_train.d_max + 1
    config.custom = True
    config.da = False
    config.alpha_grl = 1.0
    config.disc_labels = 2
    model1 = T5ForConditionalGeneration.from_pretrained('t5-large', config=config)
    model2 = T5ForConditionalGeneration.from_pretrained('t5-large', config=config)
    # tokenizer = T5Tokenizer.from_pretrained(args.model_name, tkmax_length=args.tkmax_length)


    ckpt_path = f"{cp_mod}/epoch_{epc}.pth"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model1.load_state_dict(ckpt['model_state_dict'])

    ckpt_path = f"{cp_br}/epoch_{epc}.pth"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model2.load_state_dict(ckpt['model_state_dict'])

    dataset = AutocompleteDataset('test.csv', tokenizer, tkmax_length=48, infer=True, pred_type="mod")
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    model1 = model1.cuda(f"cuda:{cuda_select}")
    model2 = model2.cuda(f"cuda:{cuda_select}")

    model1.eval()
    model2.eval()

    results1 = []
    ids1 = []

    results2 = []
    ids2 = []

    for batch, i in tqdm(dataloader):
        input_ids1 = batch['input_ids'].squeeze(1).to(f'cuda:{cuda_select}')
        attention_mask1 = batch['attention_mask'].squeeze(1).to(f'cuda:{cuda_select}')

        input_ids2 = batch['input_ids'].squeeze(1).to(f'cuda:{cuda_select}')
        attention_mask2 = batch['attention_mask'].squeeze(1).to(f'cuda:{cuda_select}')
        with torch.no_grad():
            output1 = model1.generate(input_ids=input_ids1,
                                    attention_mask=attention_mask1, 
                                    # max_length=17, 
                                    num_beams=beam, 
                                    early_stopping=True, 
                                    prefix_allowed_tokens_fn = lambda batch_id, input_ids: dataset_train.sub_get(input_ids.tolist())
                                    )
            output2 = model2.generate(input_ids=input_ids2,
                                    attention_mask=attention_mask2, 
                                    # max_length=17, 
                                    num_beams=beam, 
                                    early_stopping=True, 
                                    prefix_allowed_tokens_fn = lambda batch_id, input_ids: dataset_train_br.sub_get(input_ids.tolist())
                                    )
            output_text1 = output1.detach().clone().cpu()
            output_text2 = output2.detach().clone().cpu()
            results1 += [output_text1.tolist()]
            results2 += [output_text2.tolist()]
            ids1 += list(i)
            ids2 += list(i)
            torch.cuda.empty_cache()

    
    with open("results1.pkl", "wb") as f:
        pickle.dump(results1, f)
    
    with open("results1.pkl", "rb") as f:
        results1 = pickle.load(f)
    
    with open("results2.pkl", "wb") as f:
        pickle.dump(results2, f)
    
    with open("results2.pkl", "rb") as f:
        results2 = pickle.load(f)

    fin1_res = []
    for i in results1:
        fin1_res += i
    
    fin2_res = []
    for i in results2:
        fin2_res += i

    try:
        ids1 = [i.tolist() for i in ids1]
        ids2 = [i.tolist() for i in ids2]
    except:
        ids1 = [i for i in range(len(fin1_res))]
        ids2 = [i for i in range(len(fin2_res))]
    
    fin1_res2 = []
    for i in fin1_res:
        if len(i) < 7:
            i += [0]*(7-len(i))
        fin1_res2.append(i)
    
    fin2_res2 = []
    for i in fin2_res:
        if len(i) < 15:
            i += [0]*(15-len(i))
        fin2_res2.append(i)

    # with open("mod_code.pkl", "rb") as f:
    #     mod_code = pickle.load(f)
    
    # with open("br_code.pkl", "rb") as f:
    #     br_code = pickle.load(f)

    fin1_res2 = [i[1:] for i in fin1_res2]
    fin2_res2 = [i[1:] for i in fin2_res2]


    with open("mod_out_dct.pkl", "rb") as f:
        mod_out_dct = pickle.load(f)
    
    with open("br_out_dct.pkl", "rb") as f:
        br_out_dct = pickle.load(f)

    sup_res = []
    for res1, res2, i in zip(fin1_res2, fin2_res2, ids1):
        dct = mod_out_dct[dataset_train.code_doc[str(res1)]].copy()
        dct2 = br_out_dct[dataset_train_br.code_doc[str(res2)]].copy()
        try:
            mod = dct["module"]
        except:
            dct["module"] = " "
            dct["supergroup"] = " "
            dct["group"] = " "
        dct["indoml_id"] = i
        dct["brand"] = dct2["brand"]
        
        sup_res.append(dct)

    predict_filename = f'test_pq_kmeans{epc}_b{beam}.predict'
    with open(predict_filename, 'w') as f:
        for item in sup_res:
            # Write each dictionary as a JSON object in a new line
            # try:
            f.write(json.dumps(item) + '\n')
