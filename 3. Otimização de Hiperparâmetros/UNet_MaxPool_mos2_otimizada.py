import sys
import os
import pickle
import time
import numpy as np
import copy
from tqdm import tqdm
import psutil
import pandas as pd
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy, Recall, Precision, Specificity, F1Score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from unet_mos2 import UNet 

from codecarbon import EmissionsTracker

def treino(modelo, otimizador, imgs_treino, mascaras_treino, 
           imgs_val, mascaras_val, n_epocas=500): 
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
 
    criterion = nn.CrossEntropyLoss() 
 
    treino_perdas = [] 
    val_perdas = []

    acuracia_classe0_lista = []
    sensibilidade_classe0_lista = []
    especificidade_classe0_lista = []
    precisao_classe0_lista = []
    f1_classe0_lista = []

    acuracia_classe1_lista = []
    sensibilidade_classe1_lista = []
    especificidade_classe1_lista = []
    precisao_classe1_lista = []
    f1_classe1_lista = []

    acuracia_classe2_lista = []
    sensibilidade_classe2_lista = []
    especificidade_classe2_lista = []
    precisao_classe2_lista = []
    f1_classe2_lista = []

    melhor_perda = float("inf") 
    contador = 0 
    paciencia = 70 
    pesos_menor_perda = None 
 
    modelo = modelo.to(device) 
 
    dataset_treino = TensorDataset(imgs_treino, mascaras_treino) 
    dataset_val = TensorDataset(imgs_val, mascaras_val) 
 
    loader_treino = DataLoader( 
        dataset_treino, 
        batch_size=2, 
        shuffle=True 
    ) 
 
    loader_val = DataLoader( 
        dataset_val, 
        batch_size=2, 
        shuffle=True 
    )

    acuracia = Accuracy(task="multiclass", num_classes=3, average=None).to(device)
    sensibilidade = Recall(task="multiclass", num_classes=3, average=None).to(device)
    especificidade = Specificity(task="multiclass", num_classes=3, average=None).to(device)
    precisao = Precision(task="multiclass", num_classes=3, average=None).to(device)
    f1 = F1Score(task="multiclass", num_classes=3, average=None).to(device)
 
    for epoca in tqdm(range(n_epocas)):        

        modelo.train() 
 
        perda_epoca = 0 
 
        for imgs, mascaras in loader_treino: 
 
            imgs = imgs.to(device) 
            mascaras = mascaras.to(device).long()
            if mascaras.ndim == 4 and mascaras.shape[1] == 1:
                mascaras = mascaras.squeeze(1)
            
            print("Shape imagens:", imgs.shape)
            print("Shape máscaras:", mascaras.shape)
            print("Tipo máscaras:", mascaras.dtype)
            print("Valores únicos:", torch.unique(mascaras))
            print("Mínimo:", mascaras.min())
            print("Máximo:", mascaras.max())
            
            otimizador.zero_grad() 
 
            preds = modelo(imgs) 
 
            perda = criterion(preds, mascaras) 
 
            perda.backward() 
            otimizador.step() 
 
            perda_epoca += perda.item() 
 
        perda_epoca /= len(loader_treino) 
        treino_perdas.append(perda_epoca) 
 
        modelo.eval() 
 
        perda_val_epoca = 0
    
        acuracia_por_classe = []
        sensibilidade_por_classe = []
        especificidade_por_classe = []
        precisao_por_classe = []
        f1_por_classe = []

        with torch.no_grad(): 
 
            for imgs, mascaras in loader_val: 
 
                imgs = imgs.to(device) 
                mascaras = mascaras.to(device).long()
                if mascaras.ndim == 4 and mascaras.shape[1] == 1:
                    mascaras = mascaras.squeeze(1)
 
                saida = modelo(imgs)

                print("Saída:", saida.shape)
                print("Número de classes:", saida.shape[1])

                probs = torch.softmax(saida, dim=1)
                preds = torch.argmax(probs, dim=1)
 
                val_perda = criterion(saida, mascaras) 
 
                perda_val_epoca += val_perda.item()

                acuracia_por_classe.append(acuracia(preds, mascaras))
                sensibilidade_por_classe.append(sensibilidade(preds, mascaras)) 
                especificidade_por_classe.append(especificidade(preds, mascaras))
                precisao_por_classe.append(precisao(preds, mascaras))
                f1_por_classe.append(f1(preds, mascaras))
        
        acc0 = sum([a[0] for a in acuracia_por_classe])/len(acuracia_por_classe)
        acc1 = sum([a[1] for a in acuracia_por_classe])/len(acuracia_por_classe)
        acc2 = sum([a[2] for a in acuracia_por_classe])/len(acuracia_por_classe)

        sen0 = sum([s[0] for s in sensibilidade_por_classe])/len(sensibilidade_por_classe)
        sen1 = sum([s[1] for s in sensibilidade_por_classe])/len(sensibilidade_por_classe)
        sen2 = sum([s[2] for s in sensibilidade_por_classe])/len(sensibilidade_por_classe)

        esp0 = sum([e[0] for e in especificidade_por_classe])/len(especificidade_por_classe)
        esp1 = sum([e[0] for e in especificidade_por_classe])/len(especificidade_por_classe)
        esp2 = sum([e[0] for e in especificidade_por_classe])/len(especificidade_por_classe)

        pre0 = sum([p[0] for p in precisao_por_classe])/len(precisao_por_classe)
        pre1 = sum([p[0] for p in precisao_por_classe])/len(precisao_por_classe)
        pre2 = sum([p[0] for p in precisao_por_classe])/len(precisao_por_classe)

        f1_0 = sum([f[0] for f in f1_por_classe])/len(f1_por_classe)
        f1_1 = sum([f[0] for f in f1_por_classe])/len(f1_por_classe)
        f1_2 = sum([f[0] for f in f1_por_classe])/len(f1_por_classe)

        acuracia_classe0_lista.append(acc0)
        sensibilidade_classe0_lista.append(sen0)
        especificidade_classe0_lista.append(esp0)
        precisao_classe0_lista.append(pre0)
        f1_classe0_lista.append(f1_0)

        acuracia_classe1_lista.append(acc1)
        sensibilidade_classe1_lista.append(sen1)
        especificidade_classe1_lista.append(esp1)
        precisao_classe1_lista.append(pre1)
        f1_classe1_lista.append(f1_1)

        acuracia_classe2_lista.append(acc2)
        sensibilidade_classe2_lista.append(sen2)
        especificidade_classe2_lista.append(esp2)
        precisao_classe2_lista.append(pre2)
        f1_classe2_lista.append(f1_2)       

        if perda_val_epoca < melhor_perda: 
 
            melhor_perda = perda_val_epoca 
 
            pesos_menor_perda = copy.deepcopy(modelo.state_dict()) 
 
            contador = 0 
 
        else: 
 
            contador += 1 
 
            if contador >= paciencia: 
 
                print(f"Treinamento interrompido na Época {epoca}") 
 
                modelo.load_state_dict(pesos_menor_perda) 
 
                break 
    
    with open("U_Net_MaxPool_mos2", "wb") as unet:
        pickle.dump(modelo, unet)
        unet.close()
 
    dicionario = {
        "Treino": treino_perdas, 
        "Validação": val_perdas,
        "Acurácia - Classe 0": acuracia_classe0_lista,
        "Sensibilidade - Classe 0": sensibilidade_classe0_lista,
        "Especificidade - Classe 0": especificidade_classe0_lista,
        "Precisão - Classe 0": precisao_classe0_lista,
        "F1 - Classe 0": f1_classe0_lista,
        "Acurácia - Classe 1": acuracia_classe1_lista,
        "Sensibilidade - Classe 1": sensibilidade_classe1_lista,
        "Especificidade - Classe 1": especificidade_classe1_lista,
        "Precisão - Classe 1": precisao_classe1_lista,
        "F1 - Classe 1": f1_classe1_lista,
        "Acurácia - Classe 2": acuracia_classe2_lista,
        "Sensibilidade - Classe 2": sensibilidade_classe2_lista,
        "Especificidade - Classe 2": especificidade_classe2_lista,
        "Precisão - Classe 2": precisao_classe2_lista,
        "F1 - Classe 2": f1_classe2_lista,
    }
 
    df = pd.DataFrame(dicionario) 
 
    os.makedirs("dfs_perdas/mos2", exist_ok=True) 
 
    df.to_csv(
        "dfs_perdas/mos2/otimizacao1.csv",
        index=False
    )
 
    return treino_perdas, val_perdas, melhor_perda, pesos_menor_perda

with EmissionsTracker(project_name="codecarbon_unet_mos2_otimizada") as tracker:

    if __name__ == "__main__":
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Início", flush=True)

        dados = np.load(
            "../Datasets_finais/graphene_dataset0.npz"
        )


        print("Carregou os dados", flush=True)

        imagens = dados["images"].astype(np.float32)
        mascaras = dados["labels"].astype(np.float32)

        del dados


        print("Carregou imagens e rótulos", flush=True)

        print("imagens:", imagens.shape, imagens.dtype, imagens.nbytes / 1024**3, "GB")
        print("mascaras:", mascaras.shape, mascaras.dtype, mascaras.nbytes / 1024**3, "GB")

        imagens_treino, imagens_val, mascaras_treino, mascaras_val = train_test_split(
            imagens,
            mascaras,
            test_size=0.1,
            random_state=42
        )

        print("treino:", imagens_treino.nbytes / 1024**3, "GB")
        print("val:", imagens_val.nbytes / 1024**3, "GB")

        del imagens
        del mascaras

        imagens_treino = torch.from_numpy(imagens_treino)
        imagens_val = torch.from_numpy(imagens_val)

        mascaras_treino = torch.from_numpy(mascaras_treino)
        mascaras_val = torch.from_numpy(mascaras_val)

        if imagens_treino.ndim == 3:

            imagens_treino = imagens_treino.unsqueeze(1)
            imagens_val = imagens_val.unsqueeze(1)

        if mascaras_treino.ndim == 3:

            mascaras_treino = mascaras_treino.unsqueeze(1)
            mascaras_val = mascaras_val.unsqueeze(1)

        tamanho_kernel = 5
        func_ativacao = nn.ReLU6(inplace=True)

        modelo =  UNet(tamanho_kernel, func_ativacao).to(device)

        otimizador = optim.AdamW(modelo.parameters(), lr=0.0001)

        treino_perdas, val_perdas, melhor_perda, pesos_menor_perda = treino(
            modelo,
            otimizador,
            imagens_treino,
            mascaras_treino,
            imagens_val,
            mascaras_val,
        )


