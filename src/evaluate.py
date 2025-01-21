import torch

def evaluate_model(model, test_loader, criterion):
    running_test_loss = 0
    correct_test = 0
    total_test = 0
    model.eval()
    with torch.no_grad():
        for img, label in test_loader:
            output = model(img)
            running_test_loss += criterion(output, label).item()

            _, preds = output.max(1)
            correct_test += (preds == label).sum().item()
            total_test += label.size(0)

    avg_test_loss = running_test_loss / len(test_loader)
    test_accuracy = correct_test / total_test

    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    return avg_test_loss, test_accuracy



