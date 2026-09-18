from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import LeNet


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    directory = Path(__file__).resolve().parent

    transform = transforms.Compose([
        # Pad 28x28 digits to match LeNet's 32x32 input.
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(
        root=str(directory / 'data'), train=True,
        download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    net = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    epochs = 15
    net.train()
    print('Start Training...')
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        print(f'Epoch {epoch + 1}/{epochs}, loss: {running_loss / len(trainset):.4f}')

    print('Finished Training')
    save_path = directory / 'lenet_mnist.pth'
    torch.save(net.state_dict(), save_path)
    print(f'Model weights saved to {save_path}')


if __name__ == '__main__':
    train()
