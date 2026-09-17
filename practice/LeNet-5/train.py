import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from model import LeNet

def train():
    # Use the GPU if available; otherwise use the CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert images to tensors and scale pixel values to the range [-1, 1].
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load training images in shuffled groups of 4.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    net = LeNet().to(device)
    # The loss measures prediction errors; Adam updates the model's weights.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # One epoch is one pass through all training images.
    epochs = 15
    print("Start Training...")
    for epoch in range(epochs):
        running_loss = 0.0
        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            # Clear gradients from the previous batch.
            optimizer.zero_grad()

            # Predict classes and measure how far the scores are from the labels.
            output = net(inputs)
            loss = criterion(output, labels)
            # Calculate gradients, then update the weights to reduce the loss.
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Show the average batch loss for this epoch.
        print(f'Epoch {epoch + 1}/{epochs}, loss: {running_loss / len(trainloader):.4f}')

    print('Finished Training')

    # Save the learned weights so eval.py can use them later.
    save_path = './lenet_cifar10.pth'
    torch.save(net.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")

if __name__ == '__main__':
    train()
