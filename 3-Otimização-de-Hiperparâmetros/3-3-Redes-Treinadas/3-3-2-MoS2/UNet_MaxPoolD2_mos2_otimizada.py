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

from unet_multiclasse import UNet_MulticlasseD2 

from codecarbon import EmissionsTracker

def treino(modelo, otimizador, loader_treino, loader_val, n_epocas=500,
           paciencia=100, split=1):

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
    melhor_epoca = 0
    contador = 0
    pesos_menor_perda = None
    melhores_metricas = None

    historico_metricas = []

    modelo = modelo.to(device)

    acuracia = Accuracy(task="multiclass", num_classes=3, average=None).to(device)
    sensibilidade = Recall(task="multiclass", num_classes=3, average=None).to(device)
    especificidade = Specificity(task="multiclass", num_classes=3, average=None).to(device)
    precisao = Precision(task="multiclass", num_classes=3, average=None).to(device)
    f1 = F1Score(task="multiclass", num_classes=3, average=None).to(device)

    metricas = [acuracia, sensibilidade, especificidade, precisao, f1]

    for epoca in tqdm(range(n_epocas), desc=f"Treinamento - Split {split}"):

        modelo.train()
        perda_epoca = 0

        for imgs, mascaras in loader_treino:
            imgs = imgs.to(device)
            mascaras = preparar_mascaras(mascaras.to(device))

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

        for metrica in metricas:
            metrica.reset()
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    resultados_folds = []
    historicos_folds = []

    for split, (treino_idx, val_idx) in enumerate(kf.split(imgs), start=1):
        print(f"\n{'=' * 40}")
        print(f"Iniciando Split {split}/{n_splits}")
        print(f"{'=' * 40}", flush=True)

        imagens_treino = imgs[treino_idx]
        imagens_val = imgs[val_idx]
        mascaras_treino = mascaras[treino_idx]
        mascaras_val = mascaras[val_idx]

        dataset_treino = TensorDataset(imagens_treino, mascaras_treino)
        dataset_val = TensorDataset(imagens_val, mascaras_val)

        loader_treino = DataLoader(dataset_treino, batch_size=2, shuffle=True)
        loader_val = DataLoader(dataset_val, batch_size=2, shuffle=True)

        modelo_fold = copy.deepcopy(modelo).to(device)
        otimizador_fold = otimizador_classe(
            modelo_fold.parameters(),
            lr=taxa_aprendizado,
            **otimizador_kwargs,
        )

        (melhor_perda, melhor_epoca, pesos_menor_perda, melhores_metricas,
         historico_treino, historico_validacao, historico_metricas) = treino(
            modelo_fold,
            otimizador_fold,
            loader_treino,
            loader_val,
            epocas,
            paciencia,
            split,
        )

        resultados_folds.append({
            "Split": split,
            "Melhor época": melhor_epoca,
            "Melhor perda": melhor_perda,
            **(melhores_metricas or {}),
        })

        historicos_folds.append({
            "Split": split,
            "Treino": historico_treino,
            "Validação": historico_validacao,
            "Métricas": historico_metricas,
        })

        del modelo_fold, otimizador_fold, loader_treino, loader_val
        del imagens_treino, imagens_val, mascaras_treino, mascaras_val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_resultados = pd.DataFrame(resultados_folds)
    os.makedirs("dfs_vc/mos2/D2", exist_ok=True)
    df_resultados.to_csv(
        "dfs_vc/mos2/D2/df_resultados_unetd2_mos2.csv",
        index=False,
    )

    return df_resultados, historicos_folds


if __name__ == "__main__":
    with EmissionsTracker(project_name="codecarbon_unetd2_mos2_validacao_cruzada") as tracker:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Início", flush=True)

        dados = np.load("../Datasets_finais/mos2_dataset_final.npz")

        print("Carregou os dados", flush=True)

        imagens = dados["images"].astype(np.float32)
        mascaras = dados["labels"].astype(np.float32)
        del dados

        print("Carregou imagens e rótulos", flush=True)
        print("imagens:", imagens.shape, imagens.dtype, imagens.nbytes / 1024**3, "GB")
        print("mascaras:", mascaras.shape, mascaras.dtype, mascaras.nbytes / 1024**3, "GB")

        imagens = torch.from_numpy(imagens)
        mascaras = torch.from_numpy(mascaras)

        if imagens.ndim == 3:
            imagens = imagens.unsqueeze(1)
        if mascaras.ndim == 3:
            mascaras = mascaras.unsqueeze(1)

        mascaras = preparar_mascaras(mascaras)

        valores_classes = torch.unique(mascaras)
        print("Valores únicos das máscaras:", valores_classes.tolist(), flush=True)
        if not torch.all((valores_classes >= 0) & (valores_classes <= 2)):
            raise ValueError(
                "As máscaras devem conter índices de classe 0, 1 e 2. "
                f"Valores encontrados: {valores_classes.tolist()}"
            )

        tamanho_kernel = 5
        func_ativacao = nn.ReLU6(inplace=True)

        modelo = UNet_MulticlasseD2(tamanho_kernel, func_ativacao).to(device)

        df_resultados, historicos_folds = validacao_cruzada(
            imagens,
            mascaras,
            modelo,
            optim.AdamW,
            0.0001,
            epocas=500,
            paciencia=100,
            n_splits=5,
        )

        print(df_resultados)
