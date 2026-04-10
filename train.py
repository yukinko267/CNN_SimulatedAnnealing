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


# SA（10個から１個）
def train_SA_10_1(model, train_loader, device, epoch, T,
             can=CFG.can, noise_scale=CFG.noise_scale, c=CFG.c, k=1):

    start_time = time.time()

    current_w = flatten_parameters(model).to(device)
    candidates = []

    # make current
    current = {
        "name": "current",
        "vec": current_w,
        "score": None,
        "acc": None,
    }

    # current forward
    set_parameters(model, current["vec"])

    f = 0.0
    R_sample = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            o = F.one_hot(labels, num_classes=10).float()
            y = F.softmax(outputs, dim=1)

            sample_f = ((o - y) ** 2).sum(dim=1).mean()
            batch_size = labels.size(0)

            f += sample_f.item() * batch_size
            R_sample += batch_size

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

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
                o = F.one_hot(labels, num_classes=10).float()
                y = F.softmax(outputs, dim=1)

                sample_f = ((o - y) ** 2).sum(dim=1).mean()
                batch_size = labels.size(0)

                f += sample_f.item() * batch_size
                R_sample += batch_size

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += batch_size

        score = 0.5 * math.sqrt(f / R_sample)
        acc = correct / total

        candidates.append({
            "name": f"can{i}",
            "vec": candidate_w,
            "score": score,
            "acc": acc
        })

        # 候補ごとの進捗を同じ行に表示
        elapsed = time.time() - start_time
        sys.stdout.write(
            f"\rEpoch {epoch+1} | Candidate {i+1}/{can} "
            f"({100*(i+1)/can:.1f}%) "
            f"| score={score:.4f} "
            f"| acc={acc:.4f} "
            f"| elapsed={elapsed:.1f}s"
        )
        sys.stdout.flush()

    print()  # 改行

    # p(x)受理, parameter更新
    best_candidate = min(candidates, key=lambda c: c["score"])

    if best_candidate["score"] <= current["score"]:
        accepted = best_candidate
    else:
        deltaf = best_candidate["score"] - current["score"]
        p = math.exp(-deltaf / (k * T))

        if random.random() < p:
            accepted = best_candidate
        else:
            accepted = current

    set_parameters(model, accepted["vec"])

    total_time = time.time() - start_time

    print(
        f"Epoch {epoch+1} Finished | "
        f"Current score={current['score']:.4f}, acc={current['acc']:.4f} | "
        f"Best candidate={best_candidate['score']:.4f}, acc={best_candidate['acc']:.4f}"
    )
    print(
        f"Accepted: {accepted['name']} | "
        f"score={accepted['score']:.4f}, acc={accepted['acc']:.4f} | "
        f"T={T:.6f} | Total time={total_time:.1f}s"
    )

    # 温度更新
    T = c * T

    return T, accepted["score"], accepted["acc"]

def train_SA_2_1(model, train_loader, device, epoch, T,
             can=CFG.can, noise_scale=CFG.noise_scale, c=CFG.c, k=1):

    start_time = time.time()

    current_w = flatten_parameters(model).to(device)
    candidates = []

    # make current
    current = {
        "name": "current",
        "vec": current_w,
        "score": None,
        "acc": None,
    }

    # current forward
    set_parameters(model, current["vec"])

    f = 0.0
    R_sample = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            o = F.one_hot(labels, num_classes=10).float()
            y = F.softmax(outputs, dim=1)

            sample_f = ((o - y) ** 2).sum(dim=1).mean()
            batch_size = labels.size(0)

            f += sample_f.item() * batch_size
            R_sample += batch_size

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

    current["score"] = 0.5 * math.sqrt(f / R_sample)
    current["acc"] = correct / total

    # make candidate
    for i in range(can):
        noise = torch.randn_like(current_w) * noise_scale
        candidate_w = current["vec"] + noise

        set_parameters(model, candidate_w)

        f = 0.0
        R_sample = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                o = F.one_hot(labels, num_classes=10).float()
                y = F.softmax(outputs, dim=1)

                sample_f = ((o - y) ** 2).sum(dim=1).mean()
                batch_size = labels.size(0)

                f += sample_f.item() * batch_size
                R_sample += batch_size

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += batch_size

        can_score = 0.5 * math.sqrt(f / R_sample)
        can_acc = correct / total

        candidates.append({
            "name": f"can{i}",
            "vec": candidate_w,
            "score": can_score,
            "acc": can_acc
        })

        # p(x)受理, parameter更新

        deltaf = 0.0
        p = 1.0

        if can_score <= current["score"]:
            accepted = {
                "name": f"can{i}",
                "vec": candidate_w,
                "score": can_score,
                "acc": can_acc
            }
        else:
            deltaf = can_score - current["score"]
            p = math.exp(-deltaf / (k * T))

            if random.random() < p:
                accepted = {
                    "name": f"can{i}",
                    "vec": candidate_w,
                    "score": can_score,
                    "acc": can_acc
                }
            else:
                accepted = current
        
        current = accepted  # currentは常に最新の受理されたものを指すようにする
        set_parameters(model, accepted["vec"])

        # 温度更新
        T = c * T

        elapsed = time.time() - start_time
        sys.stdout.write(
            f"\rEpoch {epoch+1} | Step {i+1}/{can} "
            f"| Accepted: {accepted['name']:<7} "
            f"| score={accepted['score']:.4f} "
            f"| acc={accepted['acc']:.4f} "
            f"| T={T:.6f} "
            f"| delta={deltaf:.6f} | p={p:.4f} "
            f"| elapsed={elapsed:.1f}s"
        )
        sys.stdout.flush()
        print()

    return T, accepted["score"], accepted["acc"]

def train_sgd(model, train_loader, criterion, optimizer, device, epoch):  # 1 epoch

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    total_steps = len(train_loader)
    start_time = time.time()

    for step, (images, labels) in enumerate(train_loader, start=1):
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

        # 経過時間
        elapsed = time.time() - start_time

        # 進捗表示（同じ行を上書き）
        sys.stdout.write(
            f"\rEpoch {epoch} | Step {step}/{total_steps} "
            f"({100 * step / total_steps:.1f}%) "
            f"| Loss: {loss.item():.4f} "
            f"| Elapsed: {elapsed:.1f}s"
        )
        sys.stdout.flush()

    # epoch終了後に改行
    print()

    avg_loss = total_loss / total_steps
    accuracy = correct / total
    total_time = time.time() - start_time

    print(
        f"Epoch {epoch} Finished | "
        f"Avg Loss: {avg_loss:.4f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Total Time: {total_time:.1f}s"
    )

    return avg_loss, accuracy

def train_sgd_SA(model, train_loader, criterion, optimizer, device, epoch, T,
             can=CFG.can, noise_scale=CFG.noise_scale, c=CFG.c, k=1):
    # process of CNN
    model.train()
    T = CFG.T #エポックごとに初期化

    total_loss = 0.0
    correct = 0
    total = 0

    total_steps = len(train_loader)
    start_time = time.time()

    for step, (images, labels) in enumerate(train_loader, start=1):
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

        # 経過時間
        elapsed = time.time() - start_time

        # 進捗表示（同じ行を上書き）
        sys.stdout.write(
            f"\rEpoch {epoch} | Step {step}/{total_steps} "
            f"({100 * step / total_steps:.1f}%) "
            f"| Loss: {loss.item():.4f} "
            f"| Elapsed: {elapsed:.1f}s"
        )
        sys.stdout.flush()

    # epoch終了後に改行
    print()

    # process of SA
    start_time = time.time()

    current_w = flatten_parameters(model).to(device)
    candidates = []

    # make current
    current = {
        "name": "current",
        "vec": current_w,
        "score": None,
        "acc": None,
    }

    # current forward
    set_parameters(model, current["vec"])

    f = 0.0
    R_sample = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            o = F.one_hot(labels, num_classes=10).float()
            y = F.softmax(outputs, dim=1)

            sample_f = ((o - y) ** 2).sum(dim=1).mean()
            batch_size = labels.size(0)

            f += sample_f.item() * batch_size
            R_sample += batch_size

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

    current["score"] = 0.5 * math.sqrt(f / R_sample)
    current["acc"] = correct / total

    # make candidate
    for i in range(can):
        noise = torch.randn_like(current_w) * noise_scale
        candidate_w = current["vec"] + noise

        set_parameters(model, candidate_w)

        f = 0.0
        R_sample = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                o = F.one_hot(labels, num_classes=10).float()
                y = F.softmax(outputs, dim=1)

                sample_f = ((o - y) ** 2).sum(dim=1).mean()
                batch_size = labels.size(0)

                f += sample_f.item() * batch_size
                R_sample += batch_size

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += batch_size

        can_score = 0.5 * math.sqrt(f / R_sample)
        can_acc = correct / total

        candidates.append({
            "name": f"can{i}",
            "vec": candidate_w,
            "score": can_score,
            "acc": can_acc
        })

        # p(x)受理, parameter更新

        deltaf = 0.0
        p = 1.0

        if can_score <= current["score"]:
            accepted = {
                "name": f"can{i}",
                "vec": candidate_w,
                "score": can_score,
                "acc": can_acc
            }
        else:
            deltaf = can_score - current["score"]
            p = math.exp(-deltaf / (k * T))

            if random.random() < p:
                accepted = {
                    "name": f"can{i}",
                    "vec": candidate_w,
                    "score": can_score,
                    "acc": can_acc
                }
            else:
                accepted = current
        
        current = accepted  # currentは常に最新の受理されたものを指すようにする
        set_parameters(model, accepted["vec"])

        # 温度更新
        T = c * T

        elapsed = time.time() - start_time
        sys.stdout.write(
            f"\rEpoch {epoch+1} | Step {i+1}/{can} "
            f"| Accepted: {accepted['name']:<7} "
            f"| score={accepted['score']:.4f} "
            f"| acc={accepted['acc']:.4f} "
            f"| T={T:.6f} "
            f"| delta={deltaf:.6f} | p={p:.4f} "
            f"| elapsed={elapsed:.1f}s"
        )
        sys.stdout.flush()
        print()

    return T, accepted["score"], accepted["acc"]