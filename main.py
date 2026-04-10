from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model.model_CNN import CNN
import math
import random
import time
import sys
from config import CFG
from train import flatten_parameters, set_parameters, train_SA_2_1, train_sgd, train_SA_10_1, train_sgd_SA
from val import validate
import json
import os
from datetime import datetime



# 訓練データ全体での最適化関数を実装できてない finish
# 確率で遷移するのを実装できていない finish

# train_acc, train_lossをcurrent, candidatesすべてで計算
# これらは配列で管理
# paremetreが更新され保持されるか確認



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.ToTensor()


    # 前処理で正規化を入れるとSAが有利になる可能性あり

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    val_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False)

    images, labels = next(iter(train_loader))
    print(images.shape)  # torch.Size([64, 1, 28, 28])
    print(labels[:10])

    # Train LOOP
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss() # 論文を確認
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    results = []

    if CFG.SGD_or_SA == "SGD":
        for epoch in range(CFG.epochs):
            train_loss, train_acc = train_sgd(model, train_loader, criterion, optimizer, device, epoch)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            print(
                f"Epoch {epoch+1}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )
            print()
            results.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })


    elif CFG.SGD_or_SA == "SA":
        T = CFG.T
        for epoch in range(CFG.epochs):
            T, best_score, best_acc = train_SA_2_1(model, train_loader, device, epoch, T, can=CFG.can)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch+1}: "
                f"best_score={best_score:.4f}, best_acc={best_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )
            print()
            results.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })
    elif CFG.SGD_or_SA == "SGD_SA":
        T = CFG.T
        for epoch in range(CFG.epochs):
            T, best_score, best_acc = train_sgd_SA(model, train_loader, criterion, optimizer, device, epoch, T,
             can=CFG.can, noise_scale=CFG.noise_scale, c=CFG.c)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch+1}: "
                f"best_score={best_score:.4f}, best_acc={best_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )
            print()
            results.append({
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_acc": val_acc
            })


    save_data = {
        "config": {},
        "results": results
    }

    # CFG を config に保存
    for key, value in vars(CFG).items():
        if not key.startswith("__"):
            save_data["config"][key] = value

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{CFG.name}_{timestamp}"

    os.makedirs("results", exist_ok=True)
    json_path = f"results/{base_name}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()