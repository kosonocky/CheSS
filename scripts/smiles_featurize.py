import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pandarallel import (
    pandarallel,
)  # see https://github.com/qilingframework/qiling/commit/902e01beb94e2e27e50d1456e51e0ef99937aff1 for fix, must go into install location and change import
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from rdkit import Chem


def canon_smiles(smiles):
    try: 
        return Chem.CanonSmiles(smiles)
    except Exception as e:
        print(e)
        print(f"SETTING TO {smiles} ERROR")
        return "ERROR"

def load_model(
    model_name: str = "ChemBERTa",
):
    """
    Instantiates pre-trained model

    """
    assert isinstance(model_name, str)

    if model_name == "ChemBERTa":
        model = AutoModelForMaskedLM.from_pretrained(
            "seyonec/PubChem10M_SMILES_BPE_450k"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "seyonec/PubChem10M_SMILES_BPE_450k", model_max_length=512,
        )
    else:
        raise

    if torch.cuda.is_available():
        print('here')
        device = torch.device("cuda:0")
        print("Using GPU.")
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    model.to(device)

    return model, device, tokenizer


def create_dataloaders(inputs, masks, batch_size: int = 16):
    """
    Creates dataloader for pytorch model eval
    """
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    dataset = TensorDataset(
        input_tensor,
        mask_tensor,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    print("Dataloader created")
    return dataloader


def eval_model(df: pd.DataFrame, model, device, tokenizer, model_name: str = "ChemBERTa"):
    """
    Load model, tokenize, get hidden states, store in df and return
    """
    if model_name == "ChemBERTa":
        # Featureize using CLS token
        encoded_corpus = tokenizer(
            text=df["SMILES"].to_list(),
            add_special_tokens=True,
            # padding="longest",
            truncation=True,
            padding=True,
            return_attention_mask=True,
        )
        print("Data tokenized")
        input_ids = np.array(encoded_corpus["input_ids"])
        attention_masks = np.array(encoded_corpus["attention_mask"])

        dataloader = create_dataloaders(input_ids, attention_masks, batch_size=16)
        model.eval()
        for count, batch in enumerate(dataloader):
            if count % 500 == 0:
                print(f"Model eval batch {count}")
            batch_inputs, batch_masks = tuple(b.to(device) for b in batch)
            with torch.no_grad():
                outputs = model(
                    batch_inputs, attention_mask=batch_masks, output_hidden_states=True
                )
                hidden_state = outputs["hidden_states"][-1][:, 0, :]
                if count == 0:
                    cls_feature_vector_cpu = hidden_state.cpu()
                else:
                    cls_feature_vector_cpu = torch.concat(
                        [cls_feature_vector_cpu, hidden_state.cpu()]
                    )
        df["features"] = cls_feature_vector_cpu.tolist()
    else:
        raise
    return df

def main(args):
    pandarallel.initialize(progress_bar=False, use_memory_fs=False)
    data_path = "ENTER_HERE"
    save_path = Path(f"ENTER_DIR_HERE/{args.model_name}")
    save_path.mkdir(parents=True, exist_ok=True)

    # load model
    model, device, tokenizer = load_model(args.model_name)

    # load data
    with open(data_path, "r") as f:
        pubchem_strings = f.read().splitlines()
    df = pd.DataFrame(pubchem_strings, columns=["SMILES"])

    for i in range(args.start_idx, args.end_idx, args.step):
        if i + args.step < args.end_idx:
            j = i + args.step
        else:
            j = args.end_idx
        
        print(f"i: {i}, j: {j}")

        sub_df = df[i:j].reset_index(drop=True)
        sub_df = eval_model(df=sub_df, model= model, device = device, tokenizer=tokenizer, model_name = args.model_name)  # generate features
        sub_df["features"] = sub_df["features"].parallel_apply(lambda x: x/np.linalg.norm(x)) # normalize features
        sub_df[["SMILES", "features"]].to_pickle(Path(save_path, f"surechembl_{str(i).zfill(9)}_{str(j).zfill(9)}.pkl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="ChemBERTa",
        choices=["ChemBERTa"],
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=2500000
    )
    parser.add_argument(
        "--step",
        type=int,
        default=100000,
    )

    args = parser.parse_args()
    main(args)