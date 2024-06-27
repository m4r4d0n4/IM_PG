from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm
from vit_pytorch.efficient import ViT
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import torch.utils.data as data
import torchvision
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os, sys
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

torch.cuda.is_available()

# Hyperparameters:
batch_size = 64 
epochs = 200
lr = 3e-5
gamma = 0.7
seed = 142
IMG_SIZE = 128
patch_size = 16
num_classes = 2

dataset_path = "./ISIC-images"
data = pd.read_csv( "./ISIC-images/metadata.csv")
class_map = {'nevus': 0, 'melanoma': 1}
# label (0:benign, 1:melanoma)
new_data = pd.DataFrame({
    'x_col': data['isic_id'].apply(lambda x: os.path.join(dataset_path, x + '.jpg')),
    'y_col': data['diagnosis'].apply(lambda x: class_map[x] if x in class_map else 3)
})

# Seleccionar 3000 filas con 1 y 3000 filas con 0
melanoma_data = new_data[new_data['y_col'] == 1].sample(n=3000, random_state=seed)
benign_data = new_data[new_data['y_col'] == 0].sample(n=3000, random_state=seed)

# Combinar y mezclar los datos seleccionados
new_data = pd.concat([melanoma_data, benign_data]).sample(frac=1, random_state=seed).reset_index(drop=True)

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):

        self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = os.path.join(".", self.img_labels.iloc[idx, 0])
        image = read_image(img_name)
        image = transforms.ToPILImage()(image)
        label = self.img_labels.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

# Transformaciones
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Cargar el dataset completo
dataset = CustomImageDataset(new_data, dataset_path, transform=transform)

# Dividir el dataset en train, val y test
train_size = 0.70
val_size = 0.15
test_size = 0.15
train_ds, temp_ds = train_test_split(dataset, test_size=1 - train_size, shuffle=False)
val_ds, test_ds = train_test_split(temp_ds, test_size=test_size/(test_size + val_size), shuffle=False)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)
valid_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)

# Training device:
device = 'cuda'

# Linear Transformer:
efficient_transformer = Linformer(dim=IMG_SIZE, seq_len=65, depth=12, heads=8, k=64)

# Vision Transformer Model: 
model = ViT(dim=IMG_SIZE, image_size=IMG_SIZE, patch_size=patch_size, num_classes=num_classes, transformer=efficient_transformer, channels=3).to(device)

# loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Learning Rate Scheduler:
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

# Entrenar:
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for data, label in train_loader:
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            
        for data, label in valid_loader:
            
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

# Guardar Model:
PATH = "epochs"+"_"+str(epochs)+"_"+"img"+"_"+str(IMG_SIZE)+"_"+"patch"+"_"+str(patch_size)+"_"+"lr"+"_"+str(lr)+".pt"
torch.save(model.state_dict(), PATH)
# Cargar Model:
PATH = "epochs"+"_"+str(epochs)+"_"+"img"+"_"+str(IMG_SIZE)+"_"+"patch"+"_"+str(patch_size)+"_"+"lr"+"_"+str(lr)+".pt"
efficient_transformer = Linformer(dim=IMG_SIZE, seq_len=65, depth=12, heads=8, k=64)
model = ViT(dim=IMG_SIZE, image_size=IMG_SIZE, patch_size=patch_size, num_classes=num_classes, transformer=efficient_transformer, channels=3)
model.load_state_dict(torch.load(PATH))

# Rendimiento
def overall_accuracy(model, test_loader, criterion):
    

    
    y_proba = []
    y_truth = []
    test_loss = 0
    total = 0
    correct = 0
    for data in tqdm(test_loader):
        X, y = data[0].to('cpu'), data[1].to('cpu')
        output = model(X)
        test_loss += criterion(output, y.long()).item()
        for index, i in enumerate(output):
            y_proba.append(i[1])
            y_truth.append(y[index])
            if torch.argmax(i) == y[index]:
                correct+=1
            total+=1
                
    accuracy = correct/total
    
    y_proba_out = np.array([float(y_proba[i]) for i in range(len(y_proba))])
    y_truth_out = np.array([float(y_truth[i]) for i in range(len(y_truth))])
    
    return test_loss, accuracy, y_proba_out, y_truth_out


loss, acc, y_proba, y_truth = overall_accuracy(model, test_loader, criterion = nn.CrossEntropyLoss())


print(f"Accuracy: {acc}")

print(pd.value_counts(y_truth))

#Curva ROC:

def plot_ROCAUC_curve(y_truth, y_proba, fig_size):

    
    fpr, tpr, threshold = roc_curve(y_truth, y_proba)
    auc_score = roc_auc_score(y_truth, y_proba)
    txt_box = "AUC Score: " + str(round(auc_score, 4))
    plt.figure(figsize=fig_size)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1],'--')
    plt.annotate(txt_box, xy=(0.65, 0.05), xycoords='axes fraction')
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.savefig('ROC.png')
plot_ROCAUC_curve(y_truth, y_proba, (8, 8))

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

y_pred = []
y_true = []

net = model
# Iteramos los datos de test
for inputs, labels in test_loader:
        output = net(inputs) 

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) 
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) 

# Clases
classes = ('Benign', 'Melanoma')

# Matriz de confusion
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('cm.png')