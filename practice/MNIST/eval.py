from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

from model import LeNet


def evaluate():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    directory = Path(__file__).resolve().parent
    weights_path = directory / 'lenet_mnist.pth'
    if not weights_path.is_file():
        raise FileNotFoundError(f'Run train.py first to create {weights_path}')

    # Use the same padding and normalization as training.
    transform = transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    testset = torchvision.datasets.MNIST(
        root=str(directory / 'data'), train=False,
        download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    net = LeNet().to(device)
    net.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {total} test images: {100 * correct / total:.2f}%')


if __name__ == '__main__':
    evaluate()