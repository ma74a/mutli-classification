
from src.load_data_ import load_data
from src.model_ import BallsClassification
import torch.nn as nn
from torch.optim import Adam
from src.train_data import train_val_model
from utils.config import *
from utils.visualize import plot_curves
import torch


print("start loading data")
train, val, test, class_to_idx = load_data()
idx_to_class = {v:k for k,v in class_to_idx.items()}
print("ended loading data\n-----------------------------------------------\n")

# single_batch = next(iter(train))
# print(single_batch[0].shape)
# print(single_batch[1])
# img = single_batch[0][0]
# print(img.shape)

model = BallsClassification()
# # out = model(single_batch[0])
# # print(out)
# critirion = nn.CrossEntropyLoss()
# optimizer = Adam(model.parameters(), lr=LEARNING_RATE)


# print("start training")
# model, train_losses, train_accuracies, val_losses, val_accuracies = train_val_model(model=model,
#                                                                                     train_loader=train,
#                                                                                     val_loader=val,
#                                                                                     criterion=critirion,
#                                                                                     optimizer=optimizer,
#                                                                                     epochs=EPOCHS)


# plot_curves(training=train_losses, validation=val_losses)
# plot_curves(training=train_accuracies, validation=val_accuracies)

# torch.save(model.state_dict(), "./models/model.pth")
# print(class_to_idx)
# print(idx_to_class)

from src.predict import predict_image
model.load_state_dict(torch.load("models/model2.pth", weights_only=True))
image_path = "/home/mahmoud/etman/python/projects/balls_classification/test_/t2.jpg"
print(predict_image(model=model, img_path=image_path, idx_to_class=idx_to_class))
