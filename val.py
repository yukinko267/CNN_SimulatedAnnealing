import torch
import time
import sys
from config import CFG

def validate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    total_steps = len(val_loader)
    start_time = time.time()

    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            elapsed = time.time() - start_time

            # 同じ行に進捗を上書き
            sys.stdout.write(
                f"\rValidation | Step {step}/{total_steps} "
                f"({100 * step / total_steps:.1f}%) "
                f"| Loss: {loss.item():.4f} "
                f"| Elapsed: {elapsed:.1f}s"
            )
            sys.stdout.flush()

    print()

    avg_loss = total_loss / total_steps
    accuracy = correct / total
    total_time = time.time() - start_time

    print(
        f"Validation Finished | "
        f"Avg Loss: {avg_loss:.4f} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Total Time: {total_time:.1f}s"
    )

    return avg_loss, accuracy