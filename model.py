from IPython import get_ipython

get_ipython().system('pip install ipywidgets')

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from torch.utils.data import Dataset
import pandas as pd

class TrashDataSet(Dataset):
    def __init__(self, root_dir, df, transforms=None):
        # initialization: called only once during creation
        self.root_dir = root_dir
        self.df = df
        column_names = df.columns
        self.file_names = df['image'].values
        if 'class' in df.columns:
            self.targets = df['class'].values
        else:
            self.targets = None
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        image = self.file_names[index]
        img_path = "./data/images/" + self.root_dir + "/" + image

        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
                
        if self.targets is None:
            target = 0            
        else: 
            target = self.targets[index]

        return img, target




from torchvision import transforms

RGB_MEAN = (0.4914, 0.4822, 0.4465)
RGB_STD = (0.2023, 0.1994, 0.2010)

df_train = pd.read_csv('train.csv')
df_train = df_train.sample(frac=1)
df_test = pd.read_csv('test.csv')
df_test = df_test.sample(frac=1)
print(df_test)


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(RGB_MEAN, RGB_STD),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(RGB_MEAN, RGB_STD),

])


training_data = TrashDataSet('train', df_train, transforms=transform_train)
test_data = TrashDataSet('test', df_test, transforms=transform_test)


batch_size = 128
from PIL import Image

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

import torchvision.models as models

model = models.resnet18()
model.fc = nn.Linear(512, 6)

print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        y = y-1
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")



def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            y = y-1
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1" 



epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")



torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")



model = models.resnet18()
model.fc = nn.Linear(512,6)
model.load_state_dict(torch.load("model.pth"))



classes = [
    "glass",
    "paper",
    "cardboard",
    "plastic",
    "metal",
    "trash"
]

model.eval()
x, y = test_data[0][0], test_data[0][1] - 1
with torch.no_grad():
    pred = model(x.unsqueeze(0))
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')

