# %%
from torch import nn
from autobot.vision import VisionTransformer


if __name__ == "__main__":
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import torch.optim as optim

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='~/data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.CIFAR10(root='~/data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    vit = VisionTransformer((4, 4), 32, 4, 10)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vit.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = vit(inputs)
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()

            # # print statistics
            # running_loss += loss.item()
            # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}\r', end='')
            # running_loss = 0.0

            break

    print('Finished Training')
