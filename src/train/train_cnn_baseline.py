from src.models.cnn_baseline import BaselineClf
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, RMSprop, lr_scheduler


def evaluate(test_dataloader, model):
    loss = 0
    accuracy = 0
    model.eval()
    for batch in test_dataloader:
        image, label = batch
        image = model(image)
        loss += F.cross_entropy(image, label)
        x, index = torch.max(image, dim=1)
        accuracy += torch.sum(index == label).item()/len(index)
    accuracy = accuracy/len(test_dataloader)
    print("accuracy: ", accuracy)

def train_cnn(train_dataset, options, epochs=5, lr=0.0001, opt_fun=Adam):
    model = BaselineClf()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=4)
    for epoch in range(epochs):
        optimizer =opt_fun(model.parameters(), lr)
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #evaluate(train_loader, model)
    torch.save(model.state_dict(), "cnn_baseline_Adam.pt")
    return model




