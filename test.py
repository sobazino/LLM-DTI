import torch
from torch.utils.data import Dataset, DataLoader
from model import LLMModel
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_recall_curve, auc
import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(__file__)
database_file = os.path.join(current_dir, '..', 'DATABASE', 'FULL_DATABASE1.txt')
database_MolTrans = os.path.join(current_dir, '..', 'DATABASE', 'MolTrans')
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
    
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0).tolist()
    return tokenizer.decode(flat)

def Generate(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    return idx_next

def EDD(row):
    result = row.iloc[0] + ":" + row.iloc[1] + "<DTI>"
    return result

def DATADFD(DF):
    DF = DF[['Target Sequence', 'SMILES', 'Label']].copy()
    DF["Target Sequence"] = DF["Target Sequence"].str.upper()
    DF['Label'] = DF['Label'].astype(int)
    
    X = DF.drop("Label", axis=1)
    Y = DF["Label"]
    return X, Y

def test(model, device, X, Y):
    global_step = 0
    model.eval()
    all_predictions = []
    all_labels = Y.to_list()

    for Input in X:
        token_ids = Generate(
            model=model,
            idx=text_to_token_ids(Input, tokenizer).to(device),
            max_new_tokens=1,
            context_size=CONFIG["context_length"]
        )
        RES = token_ids_to_text(token_ids, tokenizer)
        all_predictions.append(int(RES) if RES in ["0", "1"] else 1)
        global_step += 1
        
    M = "OK-"
    if len(set(all_predictions)) == 1:
        M = "ER-"

    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    
    accuracy = accuracy_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_predictions)
    precisions, recalls, _ = precision_recall_curve(all_labels, all_predictions)
    aupr = auc(recalls, precisions)
    
    return accuracy, roc_auc, aupr, sensitivity, specificity, M
    
def Predict(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
    return idx_next, probas.max().item()

def onemodel(model, device, X):
    model.eval()
    token_ids, maxprobas = Predict(
        model=model,
        idx=text_to_token_ids(X, tokenizer).to(device),
        max_new_tokens=1,
        context_size=CONFIG["context_length"]
    )
    RES = token_ids_to_text(token_ids, tokenizer)
    return RES, maxprobas

def ensemblemodel(listmodel, device, X, Y):
    global_step = 0
    all_predictions = []
    all_labels = Y.to_list()

    for Input in X:
        LISTRES = []
        MAXRES = []
        for model in listmodel:
            R,V = onemodel(model, device, Input)
            LISTRES.append(R)
            MAXRES.append(V)
        
        RES = LISTRES[MAXRES.index(max(MAXRES))]
        RES = int(RES) if RES in ["0", "1"] else 1
        
        all_predictions.append(RES)
        global_step += 1
        
        if global_step % 1000 == 0:
            print(f"=========== GS: {global_step} ===========")
        
    M = "OK-"
    if len(set(all_predictions)) == 1:
        M = "ER-"

    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    
    accuracy = accuracy_score(all_labels, all_predictions)
    roc_auc = roc_auc_score(all_labels, all_predictions)
    precisions, recalls, _ = precision_recall_curve(all_labels, all_predictions)
    aupr = auc(recalls, precisions)
    
    return accuracy, roc_auc, aupr, sensitivity, specificity, M

def modelload(CONFIG, PTH):
    model = LLMModel(CONFIG)
    model = model.to(device)
    state = os.path.join(current_dir, '..', 'RES', 'LLM', 'CHECKPOINTING', PTH)
    load = torch.load(state, map_location=device)
    model.load_state_dict(load['model_state_dict'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"=========== TP: {total_params:,} ===========")
    return model

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
        TRAIN_DATABASE = DATADF(pd.read_csv(os.path.join(database_MolTrans,DD,"train.csv"))).apply(ED, axis=1).tolist()
        VAL_DATABASE = DATADF(pd.read_csv(os.path.join(database_MolTrans,DD,"val.csv"))).apply(ED, axis=1).tolist()
        TEST_DATABASE = DATADF(pd.read_csv(os.path.join(database_MolTrans,DD,"test.csv"))).apply(ED, axis=1).tolist()
        
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
        all_preds.extend(preds)
        all_targets.extend(targets)
        
    accuracy = accuracy_score(all_targets, all_preds)
    return accuracy

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

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = LLMDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader
    
if __name__ == '__main__':
    torch.manual_seed(123)
    tokenizer = Tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'=========== D: {device} ===========\n')
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
        "database": "DAVIS"
    }
    
    _, TEST_DATABASE, _ = LOADDATA(CONFIG["base"],CONFIG["database"])
    test_loader = create_dataloader(
        TEST_DATABASE,
        batch_size=2,
        max_length=CONFIG["context_length"],
        stride=CONFIG["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    print(f'=========== Test: {len(test_loader)} ===========')
    
    X, Y = DATADFD(pd.read_csv(os.path.join(database_MolTrans, CONFIG["database"], "test.csv")))
    print(f"=========== X, Y: {len(X), len(Y)} ===========")
    X = X.apply(EDD, axis=1).tolist()
    
    model1 = modelload(CONFIG, '2024_12_13_19_23_02MolTrans-DAVIS[92899].pth')
    model2 = modelload(CONFIG, '2024_12_15_13_19_18MolTrans-DAVIS[50000].pth')
    model3 = modelload(CONFIG, '2024_12_15_14_18_10MolTrans-DAVIS[100000].pth') #BEST
    model4 = modelload(CONFIG, '2024_12_15_15_20_06MolTrans-DAVIS[150000].pth')
    model5 = modelload(CONFIG, '2024_12_15_16_09_03MolTrans-DAVIS[185799].pth')
    listmodel = [model1, model2, model3, model4, model5]
    
    model1 = modelload(CONFIG, '2024_12_13_23_07_01MolTrans-BindingDB[192239].pth')
    model2 = modelload(CONFIG, '2024_12_15_16_55_07MolTrans-BindingDB[50000].pth')
    model3 = modelload(CONFIG, '2024_12_15_17_34_11MolTrans-BindingDB[100000].pth')
    model4 = modelload(CONFIG, '2024_12_15_18_13_18MolTrans-BindingDB[150000].pth')
    model5 = modelload(CONFIG, '2024_12_15_18_43_29MolTrans-BindingDB[192239].pth')
    listmodel = [model1, model2, model3, model4, model5]
    
    model1 = modelload(CONFIG, '2024_12_14_15_25_38MolTrans-BIOSNAP[195567].pth')
    model2 = modelload(CONFIG, '2024_12_14_21_51_38MolTrans-BIOSNAP[195567].pth')
    model3 = modelload(CONFIG, '2024_12_15_20_48_23MolTrans-BIOSNAP[50000].pth')
    model4 = modelload(CONFIG, '2024_12_15_22_02_03MolTrans-BIOSNAP[100000].pth')
    listmodel = [model1, model2, model3, model4]
    
    accuracy, roc_auc, aupr, sensitivity, specificity, M = ensemblemodel(listmodel , device, X, Y)
    print(f"{M} Accuracy: {accuracy:.3f}, ROC AUC: {roc_auc:.3f}, AUPR (PR-AUC): {aupr:.3f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}")
    
    for index, model in enumerate(listmodel):
        accuracy, roc_auc, aupr, sensitivity, specificity, M = test(model , device, X, Y)
        print(f"MODEL[{index+1}] - {M} Accuracy: {accuracy:.3f}, ROC AUC: {roc_auc:.3f}, AUPR (PR-AUC): {aupr:.3f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}")
    
    for index, model in enumerate(listmodel):
        accuracy = test_model(test_loader, model, device)
        print(f"MODEL[{index+1}] - Accuracy: {accuracy:.10f}")
        
    model = modelload(CONFIG, '2024_12_15_14_18_10MolTrans-DAVIS[100000].pth')
    accuracy, roc_auc, aupr, sensitivity, specificity, M = test(model , device, X, Y)
    print(f"{M} Accuracy: {accuracy:.3f}, ROC AUC: {roc_auc:.3f}, AUPR (PR-AUC): {aupr:.3f}, Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}")