import torch
from torch.utils.data import Dataset, DataLoader
from model import LLMModel,Evaluate,calc_loss_batch

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
import os

current_dir = os.path.dirname(__file__)
database_file = os.path.join(current_dir, 'FULL_DATABASE1.txt')
database_dir = os.path.join(current_dir, 'database')
res = os.path.join(current_dir, '..', 'RES')

class Tokenizer:
    def __init__(self):
        self.CHAR = { "#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6, "/": 7, ".": 8, "=": 9, "@": 10, "[": 11, "]": 12, "\\": 13, "1": 14, "2": 15, "3": 16, "4": 17, "5": 18, "6": 19, "7": 20, "8": 21, "9": 22, "0": 23, "A": 24, "B": 25, "C": 26, "D": 27, "E": 28, "F": 29, "G": 30, "H": 31, "I": 32, "J": 33, "K": 34, "L": 35, "M": 36, "N": 37, "O": 38, "P": 39, "Q": 40, "R": 41, "S": 42, "T": 43, "U": 44, "V": 45, "W": 46, "X": 47, "Y": 48, "Z": 49, "a": 50, "b": 51, "c": 52, "d": 53, "e": 54, "f": 55, "g": 56, "h": 57, "i": 58, "k": 59, "l": 60, "m": 61, "n": 62, "o": 63, "p": 64, "r": 65, "s": 66, "t": 67, "u": 68, "v": 69, "w": 70, "x": 71, "y": 72, "z": 73, ":":74 , "<DTI>":75, "<EOS>": 76, }
        self.CHAR_REVERSE = {v: k for k, v in self.CHAR.items()}
    def encode(self, text):
        encoded = []
        i = 0
        while i < len(text):
            if text[i:i + 5] == "<DTI>":
                encoded.append(self.CHAR["<DTI>"])
                i += 5
            elif text[i:i + 5] == "<EOS>":
                encoded.append(self.CHAR["<EOS>"])
                i += 5
            elif text[i] in self.CHAR:
                encoded.append(self.CHAR[text[i]])
                i += 1
            else:
                raise ValueError(f"E {text[i]}")
        return encoded

    def decode(self, tokens):
        decoded = []
        for token in tokens:
            if token in self.CHAR_REVERSE:
                decoded.append(self.CHAR_REVERSE[token])
            else:
                raise ValueError(f"E {token}")
        return ''.join(decoded)

class LLMDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def pltshow(tokens_seen, train_losses, val_losses , global_step, File):
    epochs_np = np.linspace(0, global_step, len(train_losses))
    sns.set(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax1.plot(epochs_np, train_losses, linestyle="-", marker='o', color='green', label="Training Loss", linewidth=2, markersize=0)
    ax1.plot(epochs_np, val_losses, linestyle="--", marker='o', color='black', label="Validation loss", linewidth=2, markersize=0)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Steps", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Loss", fontsize=16, fontweight='bold')
    ax1.legend(loc="upper right", fontsize=14, frameon=False)
    ax1.grid(True, linestyle='--', linewidth=1.0, alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.5)
    ax1.spines['bottom'].set_linewidth(1.5)
    ax1.tick_params(axis='x', which='major', labelsize=14, width=1.5)
    ax1.tick_params(axis='y', which='major', labelsize=14, width=1.5)
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen", fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(res, "LLM", "LOSS", File))

def CD(D):
    D = ''.join(D)
    return D.replace("\n", "")

def ED(row):
    separators = [':', '<DTI>']
    result = ''
    for i, value in enumerate(row.astype(str)):
        result += value
        if i < len(row) - 1:
            result += separators[min(i, len(separators) - 1)]
    return result + "<EOS>"

def DATADF(DF):
    DF = DF[['Target Sequence', 'SMILES', 'Label']].copy()
    DF["Target Sequence"] = DF["Target Sequence"].str.upper()
    DF['Label'] = DF['Label'].astype(int)
    return DF

def LOADDATA(CC, DD):
    if CC == "FULL":
        with open(database_file, "r", encoding="utf-8") as f:
            FULL_DATABASE = f.readlines()
        print(f"=========== LINE: {len(FULL_DATABASE)} ===========\n")

        TEST_DATABASE = FULL_DATABASE[len(FULL_DATABASE)-20000:]
        VAL_DATABASE = TEST_DATABASE[:10000]
        TEST_DATABASE = TEST_DATABASE[10000:]
        TRAIN_DATABASE = FULL_DATABASE[:len(FULL_DATABASE)-20000]
    
    elif CC == "MolTrans":
        TRAIN_DATABASE = DATADF(pd.read_csv(os.path.join(database_dir,DD,"train.csv"))).apply(ED, axis=1).tolist()
        VAL_DATABASE = DATADF(pd.read_csv(os.path.join(database_dir,DD,"val.csv"))).apply(ED, axis=1).tolist()
        TEST_DATABASE = DATADF(pd.read_csv(os.path.join(database_dir,DD,"test.csv"))).apply(ED, axis=1).tolist()
        print(TRAIN_DATABASE[0])
        
    print(f"=========== FL: {len(TRAIN_DATABASE)}")
    print(f"=========== TL: {len(TEST_DATABASE)}")
    print(f"=========== VL: {len(VAL_DATABASE)}")
    
    TRAIN_DATABASE = CD(TRAIN_DATABASE)
    TEST_DATABASE = CD(TEST_DATABASE)
    VAL_DATABASE = CD(VAL_DATABASE)
    
    print(f"=========== FD: {len(TRAIN_DATABASE)}")
    print(f"=========== TD: {len(TEST_DATABASE)}")
    print(f"=========== VD: {len(VAL_DATABASE)}\n")
    return TRAIN_DATABASE, TEST_DATABASE, VAL_DATABASE

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat)

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = LLMDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, eval_freq, eval_iter):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    model.train()
    for _ in range(20):
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            scheduler.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = Evaluate(model, train_loader, val_loader, device, eval_iter, global_step % len(train_loader))
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                for param_group in optimizer.param_groups:
                    lrnum = param_group['lr']
                print(f"(Step {global_step:06d}): Train loss {train_loss:.3f}, {loss:.3f}, Val loss {val_loss:.3f}, Learning Rate: {lrnum:.10f}")

            if global_step > 0 and global_step % 50000 == 0:
                time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step,
                }
                torch.save(checkpoint, os.path.join(res, "LLM", "CHECKPOINTING", f"{time}[{global_step}].pth"))
            
    return train_losses, val_losses, track_tokens_seen , global_step

def test_model(data_loader, model, device, num_batches=None):
    all_preds = []
    all_targets = []
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        preds = torch.argmax(logits, dim=-1).flatten().cpu().numpy()
        targets = target_batch.flatten().cpu().numpy()
        print(accuracy_score(targets, preds))
        all_preds.extend(preds)
        all_targets.extend(targets)
        
    accuracy = accuracy_score(all_targets, all_preds)
    return accuracy

if __name__ == '__main__':
    torch.manual_seed(123)
    CONFIG = {
        "vocab_size": 77,
        "context_length": 512,
        "emb_dim": 384,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True,
        "eval_freq": 500, 
        "eval_iter": 5,
        "base": "MolTrans",
        "database": "BIOSNAP"
    }
    
    TRAIN_DATABASE, TEST_DATABASE, VAL_DATABASE = LOADDATA(CONFIG["base"],CONFIG["database"])
    tokenizer = Tokenizer()
    
    train_loader = create_dataloader(
        TRAIN_DATABASE,
        batch_size=2,
        max_length=CONFIG["context_length"],
        stride=CONFIG["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader(
        VAL_DATABASE,
        batch_size=2,
        max_length=CONFIG["context_length"],
        stride=CONFIG["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    test_loader = create_dataloader(
        TEST_DATABASE,
        batch_size=2,
        max_length=CONFIG["context_length"],
        stride=CONFIG["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'=========== Train: {len(train_loader)}, Validation: {len(val_loader)}, Test: {len(test_loader)}')
    print(f'=========== D: {device} ===========\n')
    model = LLMModel(CONFIG)
    
    SL = len(train_loader)*2*20
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SL)
    
    state = os.path.join(current_dir, '..', 'RES', 'LLM', 'CHECKPOINTING', '2024_12_14_15_25_38MolTrans-BIOSNAP[195567].pth')
    load = torch.load(state, map_location=device)
    model.load_state_dict(load['model_state_dict'])
    # optimizer.load_state_dict(load['optimizer_state_dict'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"=========== TP: {total_params:,} ===========")
    print(f'=========== L: {SL} ===========\n')
    train_losses, val_losses, tokens_seen , global_step = train_model(model, train_loader, val_loader, optimizer, scheduler, device, CONFIG["eval_freq"], CONFIG["eval_iter"])

    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    File = f"{time}{CONFIG['base']}-{CONFIG['database']}.pdf"
    pltshow(tokens_seen, train_losses, val_losses, global_step, File)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
    }
    File = f"{time}{CONFIG['base']}-{CONFIG['database']}[{global_step}].pth"
    torch.save(checkpoint, os.path.join(res, "LLM", "CHECKPOINTING", File))
    print(f'=========== SAVE: RES/LLM/CHECKPOINTING/{File} ===========')
    
    accuracy = test_model(test_loader, model, device, num_batches=1)
    print(f"=========== Accuracy: {accuracy:.10f} ===========")