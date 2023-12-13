from torchvision import transforms, datasets
from torch.utils.data import DataLoader

mean1=[0.485, 0.456, 0.406]
std1=[0.229, 0.224, 0.225]
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean1, std1),
])

train_dataset = datasets.ImageFolder(root='train', transform=data_transform)

data_transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean1, std1),
])

val_dataset= datasets.ImageFolder(root='valid', transform=data_transform1)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
