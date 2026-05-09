import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
import numpy as np
from torch import nn, optim
from torch.nn import Linear
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from collections import OrderedDict
from torch.autograd import Variable
from PIL import Image
from torch.optim import lr_scheduler
import copy
import pandas as pd
import json
import os
from os.path import exists
from sklearn.metrics import accuracy_score, roc_auc_score, auc, precision_recall_curve
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, plot_confusion_matrix
import random

# Verificando se existe uma GPU dedicada
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#Diretorios do dataset
data_dir = "/home/larissa/Dev/TCC-Erik/5-fold/dataset/round_6/"
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

nThreads = 4
batch_size = 23

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

use_gpu = torch.cuda.is_available()
n_classes = 4
num_epochs = 30
lr = 0.0019439760926389421
mmt = 0.9345560727043047

model_name = "resnet50"

name = './Resultados_Artigo'
if os.path.isdir(name) == False:
    os.mkdir(name)

resultados_dir = './Resultados_Artigo/'+model_name
if os.path.isdir(resultados_dir) == False:
    os.mkdir(resultados_dir)


# Definindo as transformações para o treino e teste
# Aumento de dados e normalização
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size=[224,224]),
        transforms.RandomRotation(0,360),
        #transforms.RandomRotation(30),
        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=[224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=[224,224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Carrega o dataset do diretório
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid', 'test']}

# Definindo os dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'valid', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}

# Salvando as classes para utilizar na matriz de correlação
classes = []
for i in range(len(dataloaders['valid'].dataset.classes)):
    classes.append(dataloaders['valid'].dataset.classes[i])


# Construção do modelo

#model = models.squeezenet1_0(pretrained=True)
#model = models.alexnet(pretrained=True)
#model = models.densenet121(pretrained=True)
model = models.resnet50(pretrained = True)

# Isolando a parte final da rede pre-treinada
for param in model.parameters():
    param.requires_grad = False

# Arquitetura do modelo
#print(model)

# Definindo a parte final e classificação da rede
from collections import OrderedDict

# Criando o classificador
#model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1,1), stride=(1,1)) #SqueezeNet
#model.classifier[6] = nn.Linear(4096,n_classes)
model.fc = nn.Linear(2048, n_classes) #ResNet-50

print(model)


# Treino do modelo
def train_model(model, criterion, optimizer, num_epochs=num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Cada epoca tem uma fase de treino e validação
        for phase in ['train', 'valid']:
            if phase == 'train': 
                model.train()  # Chamando a fase de treinamento
            else:
                model.eval()   # Chamando a fase de validação

            running_loss = 0.0
            running_corrects = 0

            # Iteração dos dados.
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid accuracy: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


# Treino do modelo com a rede pre-treinada


if use_gpu:
    print ("Using GPU: "+ str(use_gpu))
    model = model.cuda()

# NLLLoss pois a saida é LogSoftmax
criterion = nn.CrossEntropyLoss()

# Otimizador
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mmt)
# Decay LR by a factor of 0.1 every 5 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


model_ft = train_model(model, criterion, optimizer, num_epochs=num_epochs)


def pytorch_predict(model, test_loader, device):
    '''
    Make prediction from a pytorch model 
    '''
    # Avaliação do modelo
    model.eval()
    
    y_true = torch.tensor([], dtype=torch.long, device=device)
    all_outputs = torch.tensor([], device=device)
    
    # deactivate autograd engine and reduce memory usage and speed up computations
    with torch.no_grad():
        for data in test_loader:
            inputs = [i.to(device) for i in data[:-1]]
            labels = data[-1].to(device)
            
            outputs = model(*inputs)
            y_true = torch.cat((y_true, labels), 0)
            all_outputs = torch.cat((all_outputs, outputs), 0)
    
    y_true = y_true.cpu().numpy()  
    _, y_pred = torch.max(all_outputs, 1)
    y_pred = y_pred.cpu().numpy()
    y_pred_prob = F.softmax(all_outputs, dim=1).cpu().numpy()
    
    return y_true, y_pred, y_pred_prob

test_loader = dataloaders['test']
#test_loader = dataloaders['valid']
y_true, y_pred, y_pred_prob = pytorch_predict(model, test_loader, device)


acuracia = accuracy_score(y_true, y_pred)
print("Acurácia:", acuracia)

report = classification_report(y_true, y_pred)
print(classification_report(y_true, y_pred))

file_report = open('./Resultados_Artigo/'+model_name+'/report_matrix.txt', 'a+')
file_report.write(report)
file_report.write("\n")
file_report.close()

cm_df = pd.DataFrame (confusion_matrix(y_true, y_pred), columns = classes)

corr = cm_df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr, mask=mask, cmap='winter', vmax=0.05, square=True)
    plt.savefig(resultados_dir+'/'+'matrizconfusao_'+'.png')
    plt.close()

# Do validation on the test set
def test(model, dataloaders, device):
  model.eval()
  accuracy = 0
  
  model.to(device)
    
  for images, labels in dataloaders['valid']:
    images = Variable(images)
    labels = Variable(labels)
    images, labels = images.to(device), labels.to(device)
      
    output = model.forward(images)
    ps = torch.exp(output)
    equality = (labels.data == ps.max(1)[1])
    accuracy += equality.type_as(torch.FloatTensor()).mean()
      
    print("Testing Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid'])))

test(model, dataloaders, device)


# Salva o estado atual do modelo 

model.class_to_idx = dataloaders['train'].dataset.class_to_idx
model.epochs = num_epochs
checkpoint = {'input_size': [3, 224, 224],
                 'batch_size': dataloaders['train'].batch_size,
                  'output_size': 39,
                  'state_dict': model.state_dict(),
                  'data_transforms': data_transforms,
                  'optimizer_dict':optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'epoch': model.epochs}
torch.save(checkpoint, resultados_dir+'/'+'model_checkpoint.pth')

# Função para o carregamento do modelo salvo anteriormente

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.resnet50()
    #model = models.squeezenet1_0()
    #model = models.alexnet()
    #model = models.densenet121
    
    # Our input_size matches the in_features of pretrained model
    input_size = 4096
    output_size = 4

    #model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1,1), stride=(1,1))   #SqueezeNet
    #model.classifier[6] = nn.Linear(4096,n_classes) # AlexNet
    model.fc = nn.Linear(2048, n_classes) # ResNet
    #model.fc = nn.Linear(2048, n_classes) # ResNet
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint['class_to_idx']

# Get index to class mapping
loaded_model, class_to_idx = load_checkpoint(resultados_dir+'/'+'model_checkpoint.pth')
idx_to_class = { v : k for k,v in class_to_idx.items()}

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model

    size = 224, 224
    image.thumbnail(size, Image.ANTIALIAS)
    image = image.crop((128 - 112, 128 - 112, 128 + 112, 128 + 112))
    npImage = np.array(image)
    npImage = npImage/255.
        
    imgA = npImage[:,:,0]
    imgB = npImage[:,:,1]
    imgC = npImage[:,:,2]
    
    imgA = (imgA - 0.485)/(0.229) 
    imgB = (imgB - 0.456)/(0.224)
    imgC = (imgC - 0.406)/(0.225)
        
    npImage[:,:,0] = imgA
    npImage[:,:,1] = imgB
    npImage[:,:,2] = imgC
    
    npImage = np.transpose(npImage, (2,0,1))
    
    return npImage

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    
    image = torch.FloatTensor([process_image(Image.open(image_path))])
    model.eval()
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]
    

    top_idx = np.argsort(pobabilities)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]

    return top_probability, top_class

# Display an image along with the top 5 classes
def view_classify(img, probabilities, classes, mapper):
    ''' Function for viewing an image and it's predicted classes.
    '''
    img_filename = img.split('/')[-2]
    img = Image.open(img)

    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
    flower_name = mapper[img_filename]
    
    ax1.set_title(flower_name)
    ax1.imshow(img)
    ax1.axis('off')
    
    y_pos = np.arange(len(probabilities))
    ax2.barh(y_pos, probabilities)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([mapper[x] for x in classes])
    ax2.invert_yaxis()

#img = test_dir + '/Soybean___healthy/018a4d64-5692-4681-b7f6-7367540946a0___RS_HL 3796.JPG'
#img = test_dir + '/Grape___Esca_(Black_Measles)/062d27c7-8c0f-4ed9-bd63-e506d9965281___FAM_B.Msls 4200.JPG'
#img = test_dir + '/Tomato___Leaf_Mold/0185befe-f0b5-4848-9677-f33c2237f4e9___Crnl_L.Mold 8729.JPG'
#img = test_dir + '/Apple___Cedar_apple_rust/025b2b9a-0ec4-4132-96ac-7f2832d0db4a___FREC_C.Rust 3655.JPG'

#p, c = predict(img, loaded_model)
#view_classify(img, p, c, cat_to_name)