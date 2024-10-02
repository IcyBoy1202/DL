pip install torch torchvision numpy matplotlib
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt
data_dir = '/content/drive/MyDrive/archive (1)/chest_xray'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}
image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val']),
    'test': datasets.ImageFolder(test_dir, data_transforms['test'])
}
batch_size = 32

dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=2),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=2),
    'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=2)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

class_names = image_datasets['train'].classes
print(class_names)
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
model = convnext_base(weights=weights)


num_ftrs = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_ftrs, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Scheduler để điều chỉnh learning rate
from torch.optim import lr_scheduler
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs -1}')
        print('-' * 10)


        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0


            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Hoàn thành huấn luyện trong {time_elapsed // 60:.0f}m {time_elapsed %60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')


    model.load_state_dict(best_model_wts)
    return model
num_epochs = 25
model = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)
model_save_path = '/content/drive/MyDrive/archive (1)/chest_xray/convnext_chest_xray.pth'

torch.save(model.state_dict(), model_save_path)
from PIL import Image
weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
model = models.convnext_base(weights=weights)

num_ftrs = model.classifier[2].in_features
model.classifier[2] = nn.Linear(num_ftrs, 2)


model_load_path = '/content/drive/MyDrive/archive (1)/chest_xray/convnext_chest_xray.pth'
model.load_state_dict(torch.load(model_load_path, map_location=device))


model.eval()
model = model.to(device)

image_path = '/content/drive/MyDrive/archive (1)/chest_xray/test/PNEUMONIA/person100_bacteria_479.jpeg'
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)
input_batch = input_batch.to(device)


with torch.no_grad():
    outputs = model(input_batch)
    _, preds = torch.max(outputs, 1)

class_names = ['NORMAL', 'PNEUMONIA']
predicted_class = class_names[preds[0]]

print(f'Ảnh được dự đoán là: {predicted_class}')


plt.imshow(image)
plt.title(f'Dự đoán: {predicted_class}')
plt.axis('off')
plt.show()
