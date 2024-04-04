import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load FashionMNIST dataset
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Define ResNet-18 model
resnet18 = models.resnet18(pretrained=True)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 10)  # 10 classes for FashionMNIST

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9

# Version 2.0: Fine-tune the model
resnet18_ft = models.resnet18(pretrained=True)
num_ftrs_ft = resnet18_ft.fc.in_features
resnet18_ft.fc = nn.Linear(num_ftrs_ft, 10)  # Change output to 10 classes

criterion_ft = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(resnet18_ft.parameters(), lr=0.001, momentum=0.9)

def fine_tune_model(model, criterion, optimizer, dataloader, num_epochs=5):
    model.train()
        for epoch in range(num_epochs):
                running_loss = 0.0
                        for i, data in enumerate(dataloader, 0):
                                    inputs, labels = data
                                                optimizer.zero_grad()
                                                            outputs = model(inputs)
                                                                        loss = criterion(outputs, labels)
                                                                                    loss.backward()
                                                                                                optimizer.step()
                                                                                                            running_loss += loss.item()
                                                                                                                    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}")

                                                                                                                    fine_tune_model(resnet18_ft, criterion_ft, optimizer_ft, trainloader)

                                                                                                                    # Save the fine-tuned model
                                                                                                                    torch.save(resnet18_ft.state_dict(), 'resnet18_fashionmnist_v2.pth'))
