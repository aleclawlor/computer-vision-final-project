


import pandas as pd
# load the data
foldername = './data'
# in pandas, "train" is called a dataframe (e.g., excel table)
train = pd.read_csv(foldername + 'train.csv')
# print out the data
print(train)


# # Problem 2. ModelOps (14 pts)

# ## Problem 2.1 Minimum Viable Product (MVP) (14 pts)

# ### (a1) [1 pt] Download Model: ResNet18



import torch
import torchvision.models as models
model = models.resnet18()


# ### (a2) [1 pt] Model surgery: change the last linear layer to predict one number instead


import torch.nn as nn
model.fc = nn.Linear(512, 6)


# ### (b) [1 pt] Define loss: Mean-squared error (MSE/L2 regression)



import torch.nn.functional as F
def criterion(y_gt, y_pred):
    return F.mse_loss(y_gt/100, torch.sigmoid(y_pred))


# ### (c) [1 pt] Define the optimizer


import torch.optim as optim

# freeze the weight for all conv layers
# only learn the last linear layer
for name,param in model.named_parameters():
    if 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

#### TODO
# Hint: copy it from pset 4
optimizer = optim.SGD(model.parameters(), lr=1e-3)


# ### (d1) [1 pt] Divide the images into train and validation


# from lab3
import numpy as np
np.random.seed(123)

def data_split(N, ratio=[8,2]):
  # generate a shuffle array
  shuffle_idx = np.arange(N)
  np.random.shuffle(shuffle_idx)
  # divide into train-val-test by the ratio
  data_split = (np.cumsum(ratio)/float(sum(ratio))*N).astype(int)
  out_idx = [None] * len(ratio)
  out_idx[0] = shuffle_idx[:data_split[0]]
  for i in range(1,len(ratio)):
    out_idx[i] = shuffle_idx[data_split[i-1] : data_split[i]]
  return out_idx  

# split the dataset into train-val split (8:2 ratio)
split_idx = data_split(len(train))
df_train = train.loc[split_idx[0]]

#### TODO
# Hint: understand what is in split_idx
df_valid = train.loc[split_idx[1]]
df_train = pd.read_csv(foldername + 'train.csv')
df_test = pd.read_csv(foldername + 'test.csv')
df_valid = pd.read_csv(foldername + 'val.csv')


# ### (d2) [3 pts] Build dataset class

from torch.utils.data import Dataset
from PIL import Image

metadata_cols = train.columns[1:-1]
 

# make a child class of PyTorch's dataset class
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
        # determine how many iterations in one epoch
        return len(self.df)
    
    def __getitem__(self, index):
        # called every time when the dataloader wants a sample
        # the dataset has a list of image file names
        # Input: dataloader provides a random index of the list
        # Output: corresponding image and meta data

        #### TODO
        image = self.file_names[index]
        idx = 0
        while ord(image[idx]) < 48 or ord(image[idx]) > 57:
            idx += 1
        type = image[:idx]
        img_path = "./data/" + type + "/" + image


        # img_path = self.root_dir + self.file_names[index] +'.jpg'
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)
        
        #### TODO
        
        if self.targets is None:
            # during deployment, df doesn't have the target value
            target = 0            
        else: 
            # otherwise, return the corresponding target value
            #### TODO
            target = self.targets[index]

        return img, target


# ### (d3) [1 pt] Build data transform


from torchvision import transforms

RGB_MEAN = (0.4914, 0.4822, 0.4465)
RGB_STD = (0.2023, 0.1994, 0.2010)

# unlike pset4 working on the 32x32 images from CIFAR10
# we here use the transforms for ImageNet challenge
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(RGB_MEAN, RGB_STD),
])

transform_test = transforms.Compose([
    #### TODO
    # hint: there are many "right" ways to do it
    # one idea is to take the center crop without randomflip, compared to transform_train
    transforms.CenterCrop(10),
    transforms.ToTensor(),
])


# ### (d4) [1 pt] Build Dataset



from torch.utils.data import DataLoader

train_dataset = TrashDataSet(foldername + 'train/', df_train, transforms=transform_train)

#### TODO
valid_dataset = TrashDataSet(foldername + 'train/', df_valid, transforms = transform_test)


# ### (e) [3 pts] Train it!
# 
# To get the point, you need to show that the loss is decreasing after a few epoches. As you experienced in Pset4, here is where you will find out potential bugs in your anwsers to previous questions.



#### nothing to change in this code block ####

class Config:  
  def __init__(self, **kwargs):
    # util
    self.batch_size = 16
    self.epochs = 0
    self.save_model_path = '' # use your google drive path to save the model
    self.log_interval = 100 # display after number of batches
    self.criterion = F.cross_entropy # loss for classification
    self.mode = 'train'
    for key, value in kwargs.items():
      setattr(self, key, value)
   
class Trainer:  
  def __init__(self, model, config, train_data = None, test_data = None):    
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.epochs = config.epochs
    self.save_model_path = config.save_model_path
    self.log_interval = config.log_interval
    self.mode = config.mode

    self.globaliter = 0
    self.train_loader = None
    self.test_loader = None
    batch_size = config.batch_size
    if self.mode == 'train': # training mode
      self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                          shuffle=True, num_workers=1)      
      #self.tb = TensorBoardColab()
      self.optimizer = config.optimizer
    
    if test_data is not None: # need evaluation
      self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                         shuffle=False, num_workers=1)
    
    self.model = model.to(self.device)
    self.criterion = config.criterion # loss function
    
                
  def train(self, epoch):  
    self.model.train()
    for batch_idx, (data, meta, target) in enumerate(self.train_loader):      
      self.globaliter += 1
      data, target = data.to(self.device), target.to(self.device)

      self.optimizer.zero_grad()
      predictions = self.model(data)

      loss = self.criterion(predictions, target)
      loss.backward()
      self.optimizer.step()

      if batch_idx % self.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(self.train_loader.dataset),
                  100. * batch_idx / len(self.train_loader), loss.item()))
        #self.tb.save_value('Train Loss', 'train_loss', self.globaliter, loss.item())
        #self.tb.flush_line('train_loss')
        
        
  def test(self, epoch, do_loss = True, return_pred = False):
    self.model.eval()
    test_loss = 0
    correct = 0
    pred = []
    with torch.no_grad():
      print('Start testing...')
      for data, meta, target in self.test_loader:
        data = data.to(self.device)
        predictions = self.model(data)
        if return_pred:
          pred.append(predictions.detach().cpu().numpy())
        if do_loss:
            target = target.to(self.device)        
            test_loss += self.criterion(predictions, target).item()*len(target)
            prediction = predictions.argmax(dim=1, keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
      if do_loss:
          test_loss /= len(self.test_loader.dataset)
          accuracy = 100. * correct / len(self.test_loader.dataset)
          print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
              test_loss, correct, len(self.test_loader.dataset), accuracy))
      """
      if self.mode == 'train': # add validation data to tensorboard
        self.tb.save_value('Validation Loss', 'val_loss', self.globaliter, test_loss)
        self.tb.flush_line('val_loss')
      """
      if return_pred:
        return np.hstack(pred)
  def main(self):
    pred = []
    if self.mode == 'train':
      for epoch in range(1, self.epochs + 1):          
          self.train(epoch)
          if self.test_loader is not None:
            # exist validation data
            self.test(epoch)
    if (self.save_model_path != ''):
        torch.save(self.model.state_dict(), self.save_model_path)
    elif self.mode == 'test':
      self.test(0)
    elif self.mode == 'deploy':          
      pred = self.test(0, False, True)
      return pred


# Let's kick off the training and hope it works!

# set of hyperparameters
train_config = Config(    
    criterion = criterion,
    save_model_path = '', # if you like, use your google drive path to save the model (mount google drive first)
    log_interval = 100, # display after number of batches
    batch_size = 16,
    optimizer = optimizer,
    epochs = 10,
)
Trainer(model, train_config, train_dataset, valid_dataset).main()


# ### (f) [1 pt] Create a submission
# You'll get the point if the code blocks below run through correctly.


df_test = pd.read_csv(foldername + 'test.csv')
test_dataset = TrashDataSet(foldername + 'test/', df_test, transforms=transform_test)

test_config = Config(mode='deploy', batch_size=8)
test_pred = Trainer(model, test_config, None, test_dataset).main()


# submission_df = pd.read_csv(foldername + 'sample_submission.csv')
# submission_df['class'] = test_pred.ravel()
# submission_df.to_csv('submission.csv', index = False)

# Summary
# submission_df.head(10)

