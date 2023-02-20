import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
import tensorboardX
from torch.utils.data import DataLoader
from data import WordleSet
from model import WordleBP
from utils import tokenlize
from skorch.classifier import NeuralNetClassifier
from sklearn.model_selection import cross_val_predict
from skorch.helper import SliceDict
import numpy as np
import sys
import joblib
from skorch.scoring import loss_scoring


def main():
    batch_size, hidden_dim, learning_rate, device_num, pred_word, epochs, ifensemble = 64, 32, 0.5, 0, "EERIE", 100, True
    model_path = "./results/model_{}e_{}.pt".format(epochs, learning_rate)
    ensemble_path = "./results/ensemble_{}e.pt".format(epochs)
    device = f'cuda:{device_num}' if torch.cuda.is_available() else "cpu"
    print("device:", device)
    indices_tensor, try_matrix, emb_dim, embedding, attribute, embedding_dim, pred_tuple = tokenlize(pred_word=pred_word)
    # print(attribute[:3])
    word_tuple = (indices_tensor, attribute)
    # print(indices_tensor.shape, attribute.shape)
    # print(try_matrix[0])
    dataset = WordleSet(word_tuple, try_matrix)
    # print(dataset[0][0],dataset[0][1])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = WordleBP(embedding, embedding_dim, attribute.size(-1) , hidden_dim)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = Adam(params=model.parameters(), lr=learning_rate)
    # print(list(model.parameters()))
    if not os.path.exists(model_path):
        model = model.to(device)
        # weight = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1]).to(device)
        X = SliceDict(indices=dataset.indices, attrs=dataset.attrs)
        y = dataset.labels
        # print(model(dataset.indices[0].to(device),dataset.attrs[0].to(device)),"\n",y[0])
        train(model, device, learning_rate, epochs, X, y)
        torch.save(model.state_dict(), model_path)
    # print(model(dataset.indices.to(device), dataset.attrs.to(device))[:10])
    else:
        model.load_state_dict(torch.load(model_path))
    pred_distribution, mc_confidence = mc_predict(model, pred_tuple)
    # print(pred_distribution.shape)

    if not os.path.exists(ensemble_path):
        n_ensemble = 5
        X = SliceDict(indices=dataset.indices, attrs=dataset.attrs)
        y = dataset.labels
        # weight = torch.tensor([0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1]).to(device)
        # model.train()
        # train(model, weight, device, learning_rate, epochs, X, y)
        ensemble = [WordleBP(embedding, embedding_dim, attribute.size(-1) , hidden_dim) for _ in range(n_ensemble)]
        for i, model in enumerate(ensemble):
            train(model, device, learning_rate, epochs, X, y)
            ensemble[i] = model.to("cpu")
        joblib.dump(ensemble, ensemble_path)
    else:
        ensemble = joblib.load(ensemble_path)
    for i_model, model in enumerate(ensemble):
        model.eval()
        output = model(*pred_tuple)
        output = output.detach().cpu()
        if i_model == 0:
            predictions = output
        else:
            predictions = torch.cat([predictions, output], dim=0)
            # print(predictions)
    ensemble_confidence = get_confidence(predictions)

    print(torch.round(pred_distribution * 100), f"{round(mc_confidence * 100) }%", f"{round(ensemble_confidence * 100) }%")
    # epochs = 100
    # for epoch in range(epochs):
    #     losses = []
    #     for indices, attrs, labels in dataloader:
    #         indices, attrs, labels = indices.to(device), attrs.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         reg_loss, outputs = model(indices, attrs)
    #         # print(outputs[0],labels[0])
    #         # print(outputs.shape, labels.shape)
    #         loss = criterion(outputs, labels) + reg_loss
    #         loss.backward()
    #         optimizer.step()
    #         # print(loss.shape)
    #         losses.append(loss)
    #     print("Epoch {
    # net.fit(dataset.indices, dataset.attrs, dataset.labels)
    # epochs = 100
    # for epoch in range(epochs):
    #     losses = []
    #     for indices, attrs, labels in dataloader:
    #         indices, attrs, labels = indices.to(device), attrs.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         reg_loss, outputs = model(indices, attrs)
    #         # print(outputs[0],labels[0])
    #         # print(outputs.shape, labels.shape)
    #         loss = criterion(outputs, labels) + reg_loss
    #         loss.backward()
    #         optimizer.step()
    #         # print(loss.shape)
    #         losses.append(loss)
    #     print("Epoch {} loss: {}".format(epoch + 1, torch.stack(losses).mean()))


def train(model, device, learning_rate, epochs, X, y):
    model = model.to(device)
    net = NeuralNetClassifier(
        module=model,
        criterion=nn.KLDivLoss,
        # criterion__weight=torch.tensor([0.1,0.1,0.2,0.2,0.2,0.1,0.1]),
        criterion__reduction="batchmean",
        optimizer=SGD,
        optimizer__lr=learning_rate,
        optimizer__weight_decay=5e-4,
        device=device,
        train_split=None,
        max_epochs=epochs,
        # batch_size=64,
        iterator_train__shuffle=True
    )
    model.train()
    net.fit(X, y)
    # print(loss_scoring(net, X, y))


def mc_predict(model, pred_tuple):
    model.eval()
    model.to("cpu")
    pred_distribution = model(*pred_tuple)
    fwd_passes = 5
    predictions = []
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()
    for fwd_pass in range(fwd_passes):
        output = model(*pred_tuple)
        output = output.detach().cpu()
        if fwd_pass == 0:
            predictions = output
        else:
            predictions = torch.cat([predictions, output], dim=0)
    confidence = get_confidence(predictions)
    return pred_distribution, confidence


def get_confidence(predictions):
    mean_predict = torch.mean(predictions, dim=0)
    num = predictions.size(0)
    kl_sum = torch.zeros((1,))
    for i in range(num):
        kl_sum += F.kl_div(torch.log(predictions[i]), mean_predict)
    kl = kl_sum / num
    print(kl)
    confidence = torch.exp(- kl)

    return confidence.item()


if __name__ == "__main__":
    main()
