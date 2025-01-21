import torch  

def train_val_model(model, train_loader, val_loader, criterion, optimizer, epochs=10):

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, preds = output.max(1)
            correct_train += (preds == y).sum().item()
            total_train += y.size(0)

        train_loss = running_train_loss / len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        running_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for X, y in val_loader:
                output = model(X)
                loss += criterion(output, y)
                
                running_val_loss += loss.item()
                _, preds = output.max(1)
                correct_val += (preds == y).sum().item()
                total_val += y.size(0)

        val_loss = running_val_loss / len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")



    return model, train_losses, train_accuracies, val_losses, val_accuracies