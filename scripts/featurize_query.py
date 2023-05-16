import argparse
import pickle as pkl
import numpy as np
import pandas as pd
import torch
import os
import time
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
import multiprocessing as mp

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_model(
    model_name: str = "ChemBERTa",
):
    """
    Instantiates pre-trained model and sets device / parallelizes if possible

    Model name being passed to allow for more flexibility if/when more models added

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

    This code will almost certainly be different for each model
    """
    if model_name == "ChemBERTa":
        # Tokenizer *should* pad to model max length (512) and trucate if it goes longer
        # But truncation doesn't seem to work no matter what I do. It does truncate, but still causes errors down the line
        encoded_corpus = tokenizer(
            text=df["SMILES"].to_list(),
            add_special_tokens=True,
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
            if count % 10 == 0:
                print(f"Model eval batch {count}")
            batch_inputs, batch_masks = tuple(b.to(device) for b in batch)
            with torch.no_grad():
                outputs = model(
                    batch_inputs, attention_mask=batch_masks, output_hidden_states=True
                )
                hidden_state = outputs["hidden_states"][-1][:, 0, :] # NOTE CLS feature vector. Mauy be different depending on model
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

def mp_sim(f, query_vec):
    if os.path.isfile(f):
        print(f)
        df = pd.read_pickle(f)
        feature_arr = np.array(df["features"].tolist())
        df["similarity"] = np.dot(feature_arr, query_vec)
        return df[["SMILES", "similarity"]]

def main(args):
    pandarallel.initialize(progress_bar=False, use_memory_fs=False)

    model, device, tokenizer = load_model(args.model_name)


    name = "diphenylaminocarbazole"
    smiles = [
        "c1ccc(N(c2ccccc2)c2ccc3c(c2)[nH]c2ccccc23)cc1", # 2-diphenylaminocarbazole
        "c12ccccc1c1ccc(N(c3ccccc3)c3ccccc3)cc1[nH]2", # 2-diphenylaminocarbazole
        "C1=CC=C(C=C1)N(C2=CC=CC=C2)C3=CC4=C(C=C3)C5=CC=CC=C5N4", # 2-diphenylaminocarbazole
    ]





    query_df = pd.DataFrame({"SMILES": smiles})
    query_df = eval_model(df=query_df, model = model, device = device, tokenizer = tokenizer, model_name=args.model_name)  # generate feature vector
    with open(f"each_canon_query_{name}.pkl", "wb") as f:
        pkl.dump(query_df, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="ChemBERTa",
        choices=["ChemBERTa"],
    )

    args = parser.parse_args()
    main(args)