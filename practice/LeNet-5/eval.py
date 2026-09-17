import torch
import torchvision
import torchvision.transforms as transforms
from model import LeNet

def evaluate():
    # Use the GPU if available; otherwise use the CPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Prepare images in the same way as during training.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Use test images to check performance on data not used for training.
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    # Create the model and load the weights saved by train.py.
    net = LeNet().to(device)
    weights_path = './lenet_cifar10.pth'
    net.load_state_dict(torch.load(weights_path, map_location=device))

    # Put the model in evaluation mode.
    net.eval()

    correct = 0
    total = 0

    # Skip gradient calculations because we are not updating weights.
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            # Choose the class with the highest score for each image.
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Turn the fraction of correct predictions into a percentage.
    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f}%')

if __name__ == '__main__':
    evaluate()
