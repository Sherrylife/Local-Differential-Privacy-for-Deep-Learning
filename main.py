import argparse
import os
import parser
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from LATENTModel import LATENTMnistCNN, LATENTCifarCNN
from torch.utils.data import DataLoader
from dataset import load_dataset
from tqdm import tqdm
from logger import LoggerCreator

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset name')
    parser.add_argument('--alpha', type=float, default=5, help='alpha LATENT')
    parser.add_argument('--eps', type=float, default=10, help='epsilon value')
    parser.add_argument('--n', type=float, default=2,
                        help='number of bits for the whole number of the binary')
    parser.add_argument('--m', type=float, default=3,
                        help='number of bits for the fraction of the binary representation')
    parser.add_argument('--bs', type=float, default=64, help='training batch size')
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
    parser.add_argument('--device', type=str, default="cuda:0", help='cuda device')

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    dataset_name = args.dataset

    log_path = './logs'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = f"{dataset_name}_{args.alpha}_{args.m}_{args.n}"
    log_name = os.path.join(log_path, log_name)
    my_log = LoggerCreator.create_logger(log_path=log_name, logging_name="LATENT Algorithm")

    if dataset_name == 'mnist' or dataset_name == 'fashion_mnist':
        model_type = LATENTMnistCNN
        input_size = 784
        output_size = 10
    elif dataset_name == 'cifar10':
        model_type = LATENTCifarCNN
        input_size = 3
        output_size = 10
    else:
        raise NotImplementedError("dataset not implemented")


    train_set, test_set = load_dataset(dataset_name)

    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

    model = model_type(
        data_in=input_size,
        data_out=output_size,
        n=args.n,
        m=args.m,
        epsilon=args.eps,
        alpha=args.alpha,
        device=args.device,
    )
    model = model.to(args.device)

    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    model.train()
    total_step = int(len(train_loader) * args.epochs)
    progress_bar = tqdm(range(total_step))

    for epoch in range(args.epochs):
        model.train()
        for idx, data in enumerate(train_loader):
            input, label = data[0].to(args.device), data[1].to(args.device)
            optimizer.zero_grad()
            output = model(input)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            my_log.info(f"epoch: {epoch}, step: {idx}, train loss: {loss.item()}")
        # evaluation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for data in test_loader:
                input, label = data[0].to(args.device), data[1].to(args.device)
                output = model(input)
                loss = F.cross_entropy(output, label)
                pred = output.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += len(label)
            acc = correct / total
        my_log.info(f"epoch: {epoch}, test accuracy: {acc}")

    my_log.info("done!")




