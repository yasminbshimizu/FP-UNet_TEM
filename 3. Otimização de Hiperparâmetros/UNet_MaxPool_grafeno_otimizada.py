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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from unet import UNet 

from codecarbon import EmissionsTracker

def treino(modelo, otimizador, imgs_treino, mascaras_treino, 
           imgs_val, mascaras_val, n_epocas=300): 
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
 
    criterion = nn.BCELoss() 
 
    treino_perdas = [] 
    val_perdas = []

    acuracias = []
    sensibilidades = []
    especificidades = []
    precisao = []
    f1 = []
    dice = []
 
    melhor_perda = float("inf") 
    contador = 0 
    paciencia = 50 
    pesos_menor_perda = None 
 
    modelo = modelo.to(device) 
 
    dataset_treino = TensorDataset(imgs_treino, mascaras_treino) 
    dataset_val = TensorDataset(imgs_val, mascaras_val) 
 
    loader_treino = DataLoader( 
        dataset_treino, 
        batch_size=8, 
        shuffle=True 
    ) 
 
    loader_val = DataLoader( 
        dataset_val, 
        batch_size=8, 
        shuffle=False 
    ) 
 
    for epoca in tqdm(range(n_epocas)): 
 
        modelo.train() 
 
        perda_epoca = 0 
 
        for imgs, mascaras in loader_treino: 
 
            imgs = imgs.to(device) 
            mascaras = mascaras.to(device) 
 
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

        verdadeiros_positivos = 0
        verdadeiros_negativos = 0
        falsos_positivos = 0
        falsos_negativos = 0
 
        with torch.no_grad(): 
 
            for imgs, mascaras in loader_val: 
 
                imgs = imgs.to(device) 
                mascaras = mascaras.to(device) 
 
                preds = modelo(imgs) 
 
                val_perda = criterion(preds, mascaras) 
 
                perda_val_epoca += val_perda.item()

                preds_binarios = (preds >= 0.5).float()

                verdadeiros_positivos += (
                    (preds_binarios == 1) & (mascaras == 1)
                ).sum().item()

                verdadeiros_negativos += (
                    (preds_binarios == 0) & (mascaras == 0)
                ).sum().item()

                falsos_positivos += (
                    (preds_binarios == 1) & (mascaras == 0)
                ).sum().item()

                falsos_negativos += (
                    (preds_binarios == 0) & (mascaras == 1)
                ).sum().item()
 
        perda_val_epoca /= len(loader_val) 
        val_perdas.append(perda_val_epoca)

        tp = verdadeiros_positivos
        tn = verdadeiros_negativos
        fp = falsos_positivos
        fn = falsos_negativos

        acuracia = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        sensibilidade = tp / (tp + fn + 1e-8)

        especificidade = tn / (tn + fp + 1e-8)

        precisao_epoca = tp / (tp + fp + 1e-8)

        f1_epoca = (
            2 * tp
            / (2 * tp + fp + fn + 1e-8)
        )

        dice_epoca = (
            2 * tp
            / (2 * tp + fp + fn + 1e-8)
        )

        acuracias.append(acuracia)
        sensibilidades.append(sensibilidade)
        especificidades.append(especificidade)
        precisao.append(precisao_epoca)
        f1.append(f1_epoca)
        dice.append(dice_epoca)
 
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
    
    with open("U_Net_MaxPool_grafeno", "wb") as unet:
        pickle.dump(modelo, unet)
        unet.close()

    dicionario = {
        "Treino": treino_perdas, 
        "Validação": val_perdas,
        "Acurácia": acuracias,
        "Recall": sensibilidades,
        "Especificidade": especificidades,
        "Precisão": precisao,
        "F1": f1,
        "Dice": dice
    }
 
    df = pd.DataFrame(dicionario) 
 
    os.makedirs("dfs_perdas/grafeno", exist_ok=True) 
 
    df.to_csv(
        "dfs_perdas/grafeno/otimizacao1.csv",
        index=False
    )
 
    os.makedirs("metricas/grafeno/u_net", exist_ok=True)
 
    return treino_perdas, val_perdas, melhor_perda, pesos_menor_perda

with EmissionsTracker(project_name="codecarbon_unet_grafeno_otimizada") as tracker:

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
        func_ativacao = nn.SiLU(inplace=True)
 
        modelo =  UNet(tamanho_kernel, func_ativacao).to(device)
        
        otimizador = optim.Adam(modelo.parameters(), lr=0.0001)
        
        treino_perdas, val_perdas, melhor_perda, pesos_menor_perda = treino(
            modelo,
            otimizador,
            imagens_treino,
            mascaras_treino,
            imagens_val,
            mascaras_val,
        )

        

