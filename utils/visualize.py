import matplotlib.pyplot as plt


def plot_curves(training, validation, title):
    plt.figure(figsize=(10, 5))
    plt.plot(training, label='Train Loss')
    plt.plot(validation, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Train vs Validation {title}')
    plt.legend()
    plt.show()