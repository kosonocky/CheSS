import argparse
import numpy as np
import pandas as pd
import torch
import os
import time

from pathlib import Path
# from pandarallel import (
#     pandarallel,
# )  # see https://github.com/qilingframework/qiling/commit/902e01beb94e2e27e50d1456e51e0ef99937aff1 for fix, must go into install location and change import
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from rdkit import Chem
import multiprocessing as mp

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def canon_smiles(smiles:str):
    """
    Canonicalize SMILES and set to "ERROR" if it cannot do it
    """
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
        # print(f)
        df = pd.read_pickle(f)
        feature_arr = np.array(df["features"].tolist())
        df["similarity"] = np.dot(feature_arr, query_vec)
        return df[["SMILES", "similarity"]]

def main(args):
    print(f"Model Name: {(model_name := args.model_name)}")
    print(f"Query SMILES: {(query_smiles := args.query_smiles)}")
    print(f"Query Name: {(query_name := args.query_name)}")
    print(f"Canonicalization: {(canonicalization := args.canonicalization)}")


    t_curr = time.time()

    model, device, tokenizer = load_model(model_name)

    # Prepare query vector
    query_df = pd.DataFrame([query_smiles], columns=["SMILES"],)

    query_df = eval_model(df=query_df, model = model, device = device, tokenizer = tokenizer, model_name=model_name)  # generate feature vector
    query_df["features"] = query_df["features"].apply(lambda x: x/np.linalg.norm(x))
    query_vec = np.array(query_df["features"].tolist())[0]

    feat_data_dir = f"ENTER_DIR_HERE"

    args1 = []
    args2 = []
    for filename in os.listdir(feat_data_dir):
        f = os.path.join(feat_data_dir, filename)
        if os.path.isfile(f):
            args1 += [f]
            args2 += [query_vec]


    n_cpus = 20
    with mp.Pool(processes= n_cpus) as pool:
        print(f"CPUs used: {n_cpus}")
        data = pool.starmap(mp_sim, zip(args1, args2))
    sim_df = pd.concat(data, axis=0)
    print(f"Total molecules searched: {len(sim_df)}")

    sim_df = sim_df.sort_values(by="similarity", ascending=False)

    if query_name is None:
        flag = query_smiles
    else:
        flag = query_name

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    sim_df.to_csv(f"{output_dir}/{flag}-{canonicalization}.csv", index=False)

    print(f"Time elapsed: {round(abs((t_old:=t_curr) - (t_curr:=time.time())), 3)} seconds.")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="ChemBERTa",
        choices=["ChemBERTa"],
    )

    parser.add_argument(
        "--query_smiles",
        type = str,
        default = "CCO"
    )

    parser.add_argument(
        "--query_name",
        type = str,
    )   

    parser.add_argument(
        "--canonicalization",
        type = str,
    )   

    args = parser.parse_args()
    main(args)