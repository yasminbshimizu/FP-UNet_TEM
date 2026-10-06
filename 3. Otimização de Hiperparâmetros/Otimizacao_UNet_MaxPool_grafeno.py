import sys
import os
import pickle
import time
import numpy as np
import copy
from tqdm import tqdm
import psutil
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import optuna 
from optuna import create_study

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from unet import UNet 

from codecarbon import EmissionsTracker


N_TENTATIVAS = 25

agora = time.localtime()
data = time.strftime("%d_%m_%Y_%H_%M_%S", agora)

processo = psutil.Process(os.getpid())

def memoria():
    print(
        f"RAM: {processo.memory_info().rss / 1024**3:.2f} GB",
        flush=True
    )

class DiceLoss:

    def __init__(self, suavizacao=1):
        self.suavizacao = suavizacao

    def __call__(self, previsao, target):
        previsao = torch.sigmoid(previsao)
        interseccao = (previsao * target).sum(dim=(2, 3))
        uniao = previsao.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * interseccao + self.suavizacao) / (uniao + self.suavizacao)
        return 1 - dice.mean()


def treino(trial, modelo, otimizador, imgs_treino, mascaras_treino,
           imgs_val, mascaras_val, n_epocas=300):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.BCELoss()

    treino_perdas = []
    val_perdas = []

    melhor_perda = float("inf")
    contador = 0
    paciencia = 50
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

        with torch.no_grad():

            for imgs, mascaras in loader_val:

                imgs = imgs.to(device)
                mascaras = mascaras.to(device)

                preds = modelo(imgs)

                val_perda = criterion(preds, mascaras)

                perda_val_epoca += val_perda.item()

        perda_val_epoca /= len(loader_val)
        val_perdas.append(perda_val_epoca)

        trial.report(perda_val_epoca, epoca)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if perda_val_epoca < melhor_perda:

            melhor_perda = perda_val_epoca

            pesos_menor_perda = copy.deepcopy(modelo.state_dict())

            contador = 0

        else:

            contador += 1

            if contador >= paciencia:

                print(f"Treinamento interrompido na época {epoca}")

                modelo.load_state_dict(pesos_menor_perda)

                break

    return treino_perdas, val_perdas, melhor_perda, pesos_menor_perda


def criar_instancia(trial, imgs_treino, mascaras_treino, imgs_val, mascaras_val):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tamanho_kernel = trial.suggest_categorical(
        "tamanho_kernel",
        [3, 5, 7]
    )

    func_ativacao = trial.suggest_categorical(
        "funcao_ativacao",
        ["relu", "relu6", "swish"]
    )

    if func_ativacao == "relu":

        func_ativacao = nn.ReLU(inplace=True)

    elif func_ativacao == "relu6":

        func_ativacao = nn.ReLU6(inplace=True)

    elif func_ativacao == "swish":
        
        func_ativacao = nn.SiLU(inplace=True)

    modelo =  UNet(tamanho_kernel, func_ativacao).to(device)

    otimizadores = trial.suggest_categorical(
        "otimizador",
        ["Adam", "AdamW", "SGD"]
    )

    taxa_aprendizado = trial.suggest_categorical(
        "taxa_aprendizado",
        [1e-2, 1e-3, 1e-4, 1e-5]
    )

    otimizador = getattr(optim, otimizadores)(
        modelo.parameters(),
        lr=taxa_aprendizado
    )

    treino_perdas, val_perdas, menor_perda, pesos_menor_perda = treino(
        trial,
        modelo,
        otimizador,
        imgs_treino,
        mascaras_treino,
        imgs_val,
        mascaras_val
    )

    dicionario = {"Treino": treino_perdas,
                  "Validação": val_perdas}
    
    df = pd.DataFrame(dicionario)

    os.makedirs("dfs_perdas/grafeno", exist_ok=True)

    df.to_csv(f"dfs_perdas/grafeno/trial_{trial.number}.csv", index=False)

    del modelo
    del otimizador
    torch.cuda.empty_cache()

    return menor_perda


def otimizacao(
    imagens_treino,
    imagens_val,
    mascaras_treino,
    mascaras_val,
    N_TENTATIVAS=100
):

    estudo_unet = create_study(
        direction="minimize",
        study_name=f"otimizacao_UNet_MaxPool_grafeno_dataset0_teste_{data}",
        storage=f"sqlite:///../Arquivos_Optuna/otimizacao_UNet_MaxPool_grafeno_dataset0_teste_{data}.db",
        load_if_exists=True,
    )

    estudo_unet.enqueue_trial({"tamanho_kernel": 3,
                         "func_ativacao": "relu",
                         "otimizador": "Adam",
                         "taxa_aprendizado": 1e-3
                         })

    estudo_unet.optimize(
        lambda trial: criar_instancia(
            trial,
            imagens_treino,
            mascaras_treino,
            imagens_val,
            mascaras_val
        ),
        n_trials=N_TENTATIVAS
    )

    melhor_tentativa = estudo_unet.best_trial
    parametros_melhor_tentativa = melhor_tentativa.params

    return parametros_melhor_tentativa


with EmissionsTracker(project_name="codecarbon_otimizacao_unet_grafeno") as tracker:

    if __name__ == "__main__":
    
        memoria()

        print("Início", flush=True)

        dados = np.load(
            "../Datasets_finais/graphene_dataset0.npz"
        )

        memoria()

        print("Carregou os dados", flush=True)

        imagens = dados["images"].astype(np.float32)
        mascaras = dados["labels"].astype(np.float32)

        del dados

        memoria()

        print("Carregou imagens e rótulos", flush=True)

        print("imagens:", imagens.shape, imagens.dtype, imagens.nbytes / 1024**3, "GB")
        print("mascaras:", mascaras.shape, mascaras.dtype, mascaras.nbytes / 1024**3, "GB")

        imagens_treino, imagens_val, mascaras_treino, mascaras_val = train_test_split(
            imagens,
            mascaras,
            test_size=0.2,
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


        resultado = otimizacao(
            imagens_treino,
            imagens_val,
            mascaras_treino,
            mascaras_val,
            N_TENTATIVAS
        )

        print(resultado)
