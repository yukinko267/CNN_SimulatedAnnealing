from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import CNN
import math
import random

class CFG():
    epochs = 10
    batch_size = 64
    SGD_or_SA = "SGD" # "SGD" or "SA"
    can = 10 # only if SA
    T = 0.1 # only if SA

def flatten_parameters(model):
    params = []

    for param in model.parameters():
        params.append(param.data.view(-1))

    return torch.cat(params)

def set_parameters(model, flat_vector):
    idx = 0

    with torch.no_grad():
        for param in model.parameters():
            numel = param.numel()

            param.copy_(
                flat_vector[idx:idx + numel].view_as(param)
            )

            idx += numel

def train_sgd(model, train_loader, criterion, optimizer, device, epoch): # 1 epoch
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # prediction
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss /len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy




# 訓練データ全体での最適化関数を実装できてない finish
# 確率で遷移するのを実装できていない finish

# train_acc, train_lossをcurrent, candidatesすべてで計算
# これらは配列で管理
# paremetreが更新され保持されるか確認


# 1epoch
def train_SA(model, train_loader, device, epoch, T, can=10, noise_scale=0.01, c=0.98, k=1):
    current_w = flatten_parameters(model).to(device)

    candidates = []

    # make current
    current = {
        "name" : "current",
        "vec" : current_w,
        "score" : None,
        "acc" : None,
    }

    # current forward
    current_w = current["vec"]
    set_parameters(model, current_w)
    f = 0
    R_sample = 0 # number of sample
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            o = F.one_hot(labels, num_classes=10).float() # one-hot化
            y = F.softmax(outputs, dim=1) # softmax出力

            # fitness関数
            sample_f = ((o - y) ** 2).sum(dim=1).mean()
            batch_size = labels.size(0)
            f += sample_f.item() * batch_size
            R_sample += batch_size

            # prediction
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        current["score"] = 0.5 * math.sqrt(f / R_sample)
        current["acc"] = correct / total

    # make candidate
    for i in range(can):
        noise = torch.randn_like(current_w) * noise_scale
        candidate_w = current_w + noise

        set_parameters(model, candidate_w)
        f = 0.0
        R_sample = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                o = F.one_hot(labels, num_classes=10).float() # one-hot化
                y = F.softmax(outputs, dim=1) # softmax出力

                # fitness関数
                sample_f = ((o - y) ** 2).sum(dim=1).mean()
                batch_size = labels.size(0)
                f += sample_f.item() * batch_size
                R_sample += batch_size

                # prediction
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        score = 0.5 * math.sqrt(f / R_sample)
        acc = correct / total

        candidates.append({
            "name" : f"can{i}",
            "vec" : candidate_w,
            "score" : score,
            "acc" : acc
        })

    # p(x)受理, parameter更新
    best_candidate = min(candidates, key=lambda c: c["score"])

    if best_candidate["score"] <= current["score"]:
        accepted = best_candidate
        set_parameters(model, accepted["vec"])
    else:
        deltaf = best_candidate["score"] - current["score"]
        p = math.exp(-deltaf / (k * T))

        if random.random() < p:
            accepted = best_candidate
            set_parameters(model, accepted["vec"])
        else:
            accepted = current
            set_parameters(model, accepted["vec"])

    # Result
    print(f"Epoch {epoch+1}: cur_score={current['score']:.4f}, cur_acc={current['acc']:.4f}, best_can_score={best_candidate['score']:.4f}, best_can_acc={best_candidate['acc']:.4f}")
    print(f"Accepted: {accepted['name']}, score={accepted['score']:.4f}, acc={accepted['acc']:.4f}")

    # T更新
    T = c*T
    return T,   accepted["score"], accepted["acc"]


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

    if CFG.SGD_or_SA == "SGD":
        for epoch in range(CFG.epochs):
            train_loss, train_acc = train_sgd(model, train_loader, criterion, optimizer, device, epoch)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            print(
                f"Epoch {epoch+1}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

    elif CFG.SGD_or_SA == "SA":
        T = CFG.T
        for epoch in range(CFG.epochs):
            T, best_score, best_acc = train_SA(model, train_loader, device, epoch, T, can=CFG.can)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch+1}: "
                f"best_score={best_score:.4f}, best_acc={best_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

if __name__ == "__main__":
    main()