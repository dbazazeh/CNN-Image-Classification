
# coding: utf-8

# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')

from collections import defaultdict
from IPython import display
from PIL import Image
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import models, transforms

import json
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import torch.nn.functional as F



# In[39]:


# Load ImageNet label to category name mapping.
imagenet_categories = list(json.load(open('data/imagenet_categories.json')).values())

# Load annotations file for the 100K training images.
mscoco_train = json.load(open('data/annotations/train2014.json'))
train_ids = [entry['id'] for entry in mscoco_train['images']]
train_id_to_file = {entry['id']: 'data/train2014/' + entry['file_name'] for entry in mscoco_train['images']}
category_to_name = {entry['id']: entry['name'] for entry in mscoco_train['categories']}
category_idx_to_name = [entry['name'] for entry in mscoco_train['categories']]
category_to_idx = {entry['id']: i for i,entry in enumerate(mscoco_train['categories'])}

# Load annotations file for the 100 validation images.
mscoco_val = json.load(open('data/annotations/val2014.json'))
val_ids = [entry['id'] for entry in mscoco_val['images']]
val_id_to_file = {entry['id']: 'data/val2014/' + entry['file_name'] for entry in mscoco_val['images']}

# We extract out all of the category labels for the images in the training set. We use a set to ignore 
# duplicate labels.
train_id_to_categories = defaultdict(set)
for entry in mscoco_train['annotations']:
    train_id_to_categories[entry['image_id']].add(entry['category_id'])

# We extract out all of the category labels for the images in the validation set. We use a set to ignore 
# duplicate labels.
val_id_to_categories = defaultdict(set)
for entry in mscoco_val['annotations']:
    val_id_to_categories[entry['image_id']].add(entry['category_id'])


# In[40]:


# Define a global transformer to appropriately scale images and subsequently convert them to a Tensor.
img_size = 224
loader = transforms.Compose([
  transforms.Scale(img_size),
  transforms.CenterCrop(img_size),
  transforms.ToTensor(),
]) 
def load_image(filename):
    """
    Simple function to load and preprocess the image.

    1. Open the image.
    2. Scale/crop it and convert it to a float tensor.
    3. Convert it to a variable (all inputs to PyTorch models must be variables).
    4. Add another dimension to the start of the Tensor (b/c VGG expects a batch).
    5. Move the variable onto the GPU.
    """
    image = Image.open(filename).convert('RGB')
    image_tensor = loader(image).float()
    image_var = Variable(image_tensor).unsqueeze(0)
    return image_var.cuda()

load_image('data/train2014/COCO_train2014_000000000009.jpg')


# Let us take a look at an image and its corresponding category labels. We consider the image with the id 391895 and the corresponding filename, `data/val2014/COCO_val2014_000000391895.jpg`. The image is shown below.
# 
# ![image](data/val2014/COCO_val2014_000000391895.jpg)
# 
# The following code determines the category labels for this image.

# In[5]:


for i,category in enumerate(val_id_to_categories[391895]):
    print("%d. %s" % (i, category_to_name[category]))
    


# # 1. Loading a Pre-trained Convolutional Neural Network (CNN)
# 
# We will work with the VGG-16 image classification CNN network first introduced in [Very Deep Convolutional Neural Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) by K. Simonyan and A. Zisserman.
# 
# Fairly straightforwardly, we load the pre-trained VGG model and indicate to PyTorch that we are using the model for inference rather than training.

# In[74]:


vgg_model = models.vgg16(pretrained=True).cuda()
vgg_model.eval()

# Let's see what the model looks like.
vgg_model


# # 2. Making Predictions Using VGG-16
# 
# Given the pre-trained network, we must now write the code to make predictions on the 10 validation images via a forward pass through the network. Typically the final layer of VGG-16 is a softmax layer, however the pre-trained PyTorch model that we are using does not have softmax built into the final layer (instead opting to incorporate it into the loss function) and therefore we must **manually** apply softmax to the output of the function.

# In[21]:


softmax = nn.Softmax()
for image_id in val_ids[:10]:
    # Display the image.
    print('\n')
    print('image with id:' , image_id)
    path = val_id_to_file[image_id]
    img =  Image.open(path).convert('RGB')
    plt.imshow(np.asarray(img))
    plt.show()

    # Print all of the category labels for this image.
    print('\nLabels:')
    for i,category in enumerate(val_id_to_categories[image_id]):
        print("%d. %s" % (i, category_to_name[category]))
       
    # Load/preprocess the image.
    img = load_image(val_id_to_file[image_id])

    # Run the image through the model and softmax.
    label_likelihoods = softmax(vgg_model(img)).squeeze()
    
    # Get the top 5 labels, and their corresponding likelihoods.
    probs, indices = label_likelihoods.topk(5)
 

    # Iterate and print out the predictions.
    imagenet_categories = [value for key,value in sorted(json.load(open('data/imagenet_categories.json')).items(), key=lambda t: int(t[0]))]
    print('\nPredictions:')
    for i in range(5):
        print('%d. %s (%.3f)' % (i, imagenet_categories[(indices[i]).data[0]], probs[i].data[0]))


    
        
       # print("label", imagenet_categories[int(indices[i])])
        #print("likelihood", probs[i])
    
    


# # 3. Computing Generic Visual Features using CNN
# 
# Since, rather than the output of VGG, we want a fixed sized vector representation of each image, we remove the last linear layer. The implementation of the forward function for VGG is shown below:
# 
# ```
# x = self.features(x)
# x = x.view(x.size(0), -1)
# x = self.classifier(x)
# ```
# We aim to preserve everything but the final component of the classifier, meaning we must define an alternative equivalent to `self.classifier`.

# In[8]:


# Remove the final layer of the classifier, and indicate to PyTorch that the model is being used for inference
# rather than training (most importantly, this disables dropout).
vgg_model2 = models.vgg16(pretrained=True).cuda()
vgg_model2.eval().cuda()

vgg_model2.classifier = nn.Sequential(*list(vgg_model2.classifier.children())[:-2])


for param in vgg_model2.parameters():   
    param.requires_grad = False

vgg_model2.cuda()

#


# In[33]:


# First we vectorize all of the features of training images and write the results to a file.
i = 0
#training_vectors = np.zeros((len(train_ids), 4096), dtype=np.float32)
training_vectors = np.zeros((len(train_ids), 4096), dtype=np.float32)
training_vectors = Variable(torch.FloatTensor(training_vectors))
#training_vectors = torch.zeros(2, 4096)
for image_id in train_ids:
    
    img = load_image(train_id_to_file[image_id])
    training_vectors[i] = vgg_model2(img).squeeze().cuda()
    i = i+1
    if(i%1000 == 0):
        print(i)
#print(training_vectors)    
#np.save(open('outputs/training_vectors', 'wb+'), training_vectors)
#pickle.dump({'training_vectors': training_vectors}, 
           # open('vgg16_trainfeatures.p', 'w'))
print("copying to file")
training_vectors2 = training_vectors.data.numpy()
np.save(open('outputs/training_vectors2', 'wb+'), training_vectors2)
print("done")


# In[21]:


#  we vectorize all of the features of validation images and write the results to a file.
i = 0

validation_vectors = np.zeros((100, 4096), dtype=np.float32)
validation_vectors = Variable(torch.FloatTensor(validation_vectors))

for image_id in val_ids[:100]:
    
    img = load_image(val_id_to_file[image_id])
    validation_vectors[i] = vgg_model2(img).squeeze().cuda()
    i = i+1
    
validation_vectors2 = validation_vectors.data.numpy()
np.save(open('outputs/validation_vectors', 'wb+'), validation_vectors2)
print("done")


# # 4. Visual Similarity
# 
# We now use the generated vectors, to find the closest training image for each validation image. This can easily be done by finding the training vector that minimizes the Euclidean distance for every validation image. We repeat this exercise for the first 10 images in the validation set.

# In[9]:


i = 0
validation_features = np.load('outputs/validation_vectors')
training_features = np.load('outputs/training_vectors')

for j in val_ids[:10]:
    d = validation_features[i] - training_features 
    z = np.power(d, 2)
    y = np.sum(z,1)
    x = np.power(y,0.5)
    minIndex = np.argmin(x, axis=0)
    print(minIndex)
    trainID = train_ids[minIndex]
    
    print('\n')
    
    print('Validation image sample', i)

    path1 = val_id_to_file[j]
    display.display(display.Image(path1))
    
    print('\n is most similar to')
    
    path2 = train_id_to_file[trainID]
    display.display(display.Image(path2))

    i = i+1
        


# # 5. Training a Multi-Label Classification Network
# 
# We now build a two layer classification network, which takes 4096-dimensional vectors as input and outputs the probabilities of the 80 categories present in MSCOCO. 
# 
# For this purpose, we utilize two layers (both containing sigmoid activation functions) with the hidden dimension set to 512. 

# In[41]:


class TwoLayerVGG2(nn.Module):
    def __init__(self):
        super(TwoLayerVGG2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(4096, 512),
           nn.Sigmoid(),
            nn.Linear(512, 80),
        )
      
    def forward(self, x):
        out = self.layer1(x)
        return out
    

VGG3 = TwoLayerVGG2()
criterion = nn.MultiLabelSoftMarginLoss().cuda()
optimizer = torch.optim.Adam(VGG3.parameters(), 0.001)


# In[6]:


# The output data is prepared by representing each output as a binary vector of categories

i = 0
trainingOutputs = np.zeros((len(train_ids), 80), dtype=np.float32)
for image_id in train_ids:
    c = train_id_to_categories[image_id]
    cc = np.zeros(len(c))
    convertToList = list(c)
    cc.astype(int)
    for j in range (len(c)):
        #cc[j] = category_to_idx[convertToList[j]]
        ind = category_to_idx[convertToList[j]]
        trainingOutputs[i][ind] = 1
    i = i+1

trainingInputs = np.load('outputs/training_vectors')

ValidationInputs = np.load('outputs/validation_vectors')

def train(model, learning_rate=0.001, batch_size=50, epochs=5):
    """
    Training function which takes as input a model, a learning rate and a batch size.
  
                                                                                                                                                                  r completing a full pass over the data, the function exists, and the input model will be trained.
    """
    for e in range (epochs):
        epoch_loss = 0
        batchIndex =0
        
        for i in range (int((len(train_ids))/batch_size)):
            minibatch_loss = 0
            batchInput = Variable(torch.FloatTensor(trainingInputs[batchIndex: batchIndex +batch_size])).squeeze()
            batchLabelOutput = Variable(torch.FloatTensor(trainingOutputs[batchIndex: batchIndex +batch_size])).squeeze()
           
            output = model(batchInput)
            optimizer.zero_grad()
            loss = criterion(output, batchLabelOutput)
            minibatch_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            
            #print(minibatch_loss /batch_size)
            #if ((i == (len(train_ids)/batch_size) - 1) ):
            #    print(minibatch_loss /batch_size)
            
            batchIndex = batchIndex + batch_size
            #epoch_loss += loss.data[0]
            #print(epoch_loss /len(train_ids))
   
# Finally train the model
train(VGG3)
print("done")
    


# In[34]:


# Now repeat step two using the two layer classifier.

j=0
VGG3.eval()

for image_id in val_ids[:10]:
    # Display the image.
    print('\n')
    print('image with id:' , image_id)
    path = val_id_to_file[image_id]
    img =  Image.open(path).convert('RGB')
    plt.imshow(np.asarray(img))
    plt.show()

    # Print all of the category labels for this image.
    print('\nLabels:')
    for i,category in enumerate(val_id_to_categories[image_id]):
        print("%d. %s" % (i, category_to_name[category]))

    # Run the image through the model and softmax.
    validationOutput = torch.sigmoid(VGG3(Variable(torch.FloatTensor(validation_features[j])))).squeeze()
    probs, indices = validationOutput.topk(5)
    
    # Get the top 5 labels, and their corresponding likelihoods.
    #probs, indices = label_likelihoods.topk(5)

    # Iterate and print out the predictions.
    #imagenet_categories = [value for key,value in sorted(json.load(open('data/imagenet_categories.json')).items(), key=lambda t: int(t[0]))]
    j = j +1
    print('\nPredictions:')
    for k in range(5):
    
        print('%d. %s(%.3f)' % (k, category_idx_to_name[indices[k].data[0]], probs[k].data[0]))
        
    


# #  Discussion
# The 2 layer NN performed well, considering the small number of epochs. For most images, at least half of the ground truth labels appeared in the predictions top 5. Image with id: 574769 for example had all prediction labels correct. However, the probabilities of each label was not so accurate. The most accurate label is person, and that is because most pictures had a person in them, meaning there were more training samples for this label exposed to the network during training. Rarely seen labels on the other hand, such as "Red wine", are harder to get accurately predicted due to less training.
# 
# 

# # 6. End-to-End Model Fine-tuning
# 
# Instead of training *only* the final two layers, we now create an end-to-end model and train the entire thing. 

# In[13]:


finalmodel2 = models.vgg16(pretrained=True)
finalmodel2.classifier = nn.Sequential(*list(finalmodel2.classifier.children())[:-1] + list(VGG3.layer1.children()))
criterion = nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adam(finalmodel2.parameters(), 0.0001)
finalmodel2 = finalmodel2.cuda()
finalmodel2.train()


# In[44]:


def load_image2(filename):
    image = Image.open(filename).convert('RGB')
    image_tensor = loader(image).float()
    image_var = Variable(image_tensor)
    return image_var


# In[14]:



def train2(model, learning_rate=0.0001, batch_size=50, epochs=2):

    for e in range (epochs):
        epoch_loss = 0
        batchIndex =0
        
        for i in range (int((len(train_ids))/batch_size)):
            minibatch_loss = 0
            
            imagesIn = []

            for image_id in train_ids[batchIndex: batchIndex +batch_size]:
                imagesIn.append(load_image2(train_id_to_file[image_id]))
    
            batchLabelOutput = Variable(torch.FloatTensor(trainingOutputs[batchIndex: batchIndex +batch_size])).cuda() 
            batchInput = torch.stack(imagesIn, dim = 0).cuda()
            output = model(batchInput).squeeze()
            optimizer.zero_grad()
            loss = criterion(output, batchLabelOutput)
            minibatch_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            
            batchIndex = batchIndex + batch_size
            #if (batchIndex % 30 == 0):
            #print(minibatch_loss/batch_size)
   
# Finally train the model
train2(finalmodel2)
print("done")


# In[16]:


# Now repeat step two using the two layer classifier.

j=0
finalmodel2.eval().cuda()

for image_id in val_ids[:10]:
    # Display the image.
    print('\n')
    print('image with id:' , image_id)
    path = val_id_to_file[image_id]
    img =  Image.open(path).convert('RGB')
    plt.imshow(np.asarray(img))
    plt.show()

    # Print all of the category labels for this image.
    print('\nLabels:')
    for i,category in enumerate(val_id_to_categories[image_id]):
        print("%d. %s" % (i, category_to_name[category]))

    # Load/preprocess the image.
    img = load_image(val_id_to_file[image_id])

    # Run the image through the model and softmax.
    validationOutput = torch.sigmoid(finalmodel2(img)).squeeze().cuda()
    
    
    probs, indices = validationOutput.topk(5)
    
    # Get the top 5 labels, and their corresponding likelihoods.
    #probs, indices = label_likelihoods.topk(5)

    # Iterate and print out the predictions.
    #imagenet_categories = [value for key,value in sorted(json.load(open('data/imagenet_categories.json')).items(), key=lambda t: int(t[0]))]
    j = j +1
    print('\nPredictions:')
    for k in range(5):
    
        print('%d. %s(%.3f)' % (k, category_idx_to_name[indices[k].data[0]], probs[k].data[0]))
        
    


# # Discussion
# 
# The End-End NN performed very similarly to the 2 layer NN, as expected, since it was initialized with the weighs of the VGG16 network and the 2 layer NN. However, there are slight improvements in some labels For example, in the 2nd image, the new model was able to correctly detect the cow, which was not previously detected in the 2 layer model. This means that indeed the new end-end model is learning. However, the changes are very small, and this is because we only iterated the dataset twice through the model, which is may not be enough to have a huge improvement. 

# # 7. Hyper-parameter Tuning
# 
# Now we do a grid search over the learning rate and batch size.

# In[45]:


def train_plot(model, learning_rate, batch_size, epochs):
    optimizer = torch.optim.Adam(model.parameters(),  learning_rate)
    criterion = nn.MultiLabelSoftMarginLoss().cuda()
    model = model.cuda()
    model.train()
    lasterror = 0
    r = int((len(train_ids))/batch_size)
  
    for e in range (epochs):
   
        batchIndex =0
        
        for i in range (r):
            minibatch_loss = 0
            
            imagesIn = []

            for image_id in train_ids[batchIndex: batchIndex +batch_size]:
                imagesIn.append(load_image2(train_id_to_file[image_id]))
    
            batchLabelOutput = Variable(torch.FloatTensor(trainingOutputs[batchIndex: batchIndex +batch_size])).cuda() 
            batchInput = torch.stack(imagesIn, dim = 0).cuda()
            output = model(batchInput).squeeze()
            optimizer.zero_grad()
            loss = criterion(output, batchLabelOutput)
            minibatch_loss += loss.data[0]
            loss.backward()
            optimizer.step()
            
            batchIndex = batchIndex + batch_size
            if (i == (r-1)):
                lasterror = minibatch_loss/batch_size
            
    
    return lasterror
   


# In[46]:


def validationdata_test(model):
    model.eval()
    errors = np.zeros(100)
    j = 0
        
    for image_id in val_ids[:100]:
    
        img = load_image(val_id_to_file[image_id])
        LabelOutput = Variable(torch.FloatTensor(validationOutputs[j])).cuda() 
        
        output = model(img).squeeze().cuda()
        loss = criterion(output, LabelOutput)

        errors[j] = loss.data[0]
        j = j+1   
    
    return errors


# In[47]:


def trainingdata_test(model,batch_size):
    model.eval().cuda()
    r = int((len(train_ids))/batch_size)
    batchIndex =0
    batcherrors = np.zeros(int(r/30))
    j = 0
        
    for i in range (r):
        minibatch_loss = 0
            
        imagesIn = []

        for image_id in train_ids[batchIndex: batchIndex +batch_size]:
              imagesIn.append(load_image2(train_id_to_file[image_id]))
                
        batchLabelOutput = Variable(torch.FloatTensor(trainingOutputs[batchIndex: batchIndex +batch_size])).cuda() 
        batchInput = torch.stack(imagesIn, dim = 0).cuda()
        output = model(batchInput).squeeze()            
        loss = criterion(output, batchLabelOutput)
        minibatch_loss += loss.data[0]
            
        batchIndex = batchIndex + batch_size
        if (i % 30 == 0 and (j < int(r/30))):
            batcherrors[j] = minibatch_loss/batch_size
            j = j+1
    
    return batcherrors


# In[48]:


#prepare validation set outputs
i = 0
validationOutputs = np.zeros((100, 80), dtype=np.float32)
for image_id in val_ids:
    c = val_id_to_categories[image_id]
    cc = np.zeros(len(c))
    convertToList = list(c)
    cc.astype(int)
    for j in range (len(c)):
        ind = category_to_idx[convertToList[j]]
        validationOutputs[i][ind] = 1
    i = i+1

    


# # Module 1 with learning rate = 0.0001, batch size = 50

# In[15]:


#First we optimize the learning rate, then the batch size

#module 1: learning rate = 0.0001, batch size = 50

#errors1 = train_plot(finalmodel2, 0.0001, 50, 1)
torch.save(finalmodel2.state_dict(), './module1.pth')
print("last training error is", errors1[54])


# In[24]:


#valerrors = validation(finalmodel2)
fig = plt.figure()
fig.suptitle('validation errors for 100 validation images', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)


ax.set_xlabel('image #')
ax.set_ylabel('validation error')
ax.plot(valerrors)


# In[33]:


trainerrors = trainingdata_test(finalmodel2,50)
fig = plt.figure()
fig.suptitle('training images errors', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)


ax.set_xlabel('minibatch #')
ax.set_ylabel('training images error')
ax.plot(trainerrors)


# # Module 2 with learning rate = 0.00001, batch size = 50¶

# In[16]:


Module2 = models.vgg16(pretrained=True)
Module2.classifier = nn.Sequential(*list(Module2.classifier.children())[:-1] + list(VGG3.layer1.children()))


# In[84]:


error2 = train_plot(Module2, 0.00001, 50, 1)
print("last training error is", error2)


# In[85]:


valerrors = validation(Module2)
fig = plt.figure()
fig.suptitle('validation errors for 100 validation images', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)


ax.set_xlabel('image #')
ax.set_ylabel('validation error')
ax.plot(valerrors)


# In[17]:


Module2 = torch.load('m2.pt')


# In[18]:


trainerrors = trainingdata_test(Module2,50)
fig = plt.figure()
fig.suptitle('training images errors', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)


ax.set_xlabel('minibatch #')
ax.set_ylabel('training images error')
ax.plot(trainerrors)


# In[87]:


torch.save(Module2, 'm2.pt')


# # Module 3 with learning rate = 0.0001, batch size = 25¶

# In[30]:


Module3 = models.vgg16(pretrained=True)
Module3.classifier = nn.Sequential(*list(Module3.classifier.children())[:-1] + list(VGG3.layer1.children()))


# In[31]:


error3 = train_plot(Module3, 0.0001, 25, 1)
print("last training error is", error3)


# In[37]:


valerrors = validationdata_test(Module3)
fig = plt.figure()
fig.suptitle('validation errors for 100 validation images', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)


ax.set_xlabel('image #')
ax.set_ylabel('validation error')
ax.plot(valerrors)


# In[32]:


trainerrors = trainingdata_test(Module3,25)
fig = plt.figure()
fig.suptitle('training images errors', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)


ax.set_xlabel('minibatch #')
ax.set_ylabel('training images error')
ax.plot(trainerrors)


# # Module 4 with learning rate = 0.00001, batch size = 25¶

# In[49]:


Module4 = models.vgg16(pretrained=True)
Module4.classifier = nn.Sequential(*list(Module4.classifier.children())[:-1] + list(VGG3.layer1.children()))


# In[50]:


error4 = train_plot(Module4, 0.00001, 25, 1)
print("last training error is", error4)


# In[51]:


valerrors = validationdata_test(Module4)
fig = plt.figure()
fig.suptitle('validation errors for 100 validation images', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)


ax.set_xlabel('image #')
ax.set_ylabel('validation error')
ax.plot(valerrors)


# In[52]:


trainerrors = trainingdata_test(Module4,25)
fig = plt.figure()
fig.suptitle('training images errors', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)


ax.set_xlabel('minibatch #')
ax.set_ylabel('training images error')
ax.plot(trainerrors)


# # Discussion
# 
# Out of the 4 modules, module 1 with learning rate = 0.0001 and batch size 50 had the lowest training loss, at the end of the training, with a mean loss of 0.00183. After training the model, both the testing and validation sets were passed on to the model and their losses were saved. As expected, training set error was much smaller than the validation set error, due to the fact that the model has previously seen these samples during training. The range of training set error lies between 0.005 and 0.0025. Module 3 and 4 with the smaller batch size showed higher training set error, compared to module 1 and 2. The validation set error was calculated for each image in the set and was in the range between 0 - 0.30, which means that it has high variation. Module 2 looks like it has an overall lower validation set error, and for this reason, my final choice of hyperparameters is a learning rate of 0.00001 and batch size of 50.

# Note: Plotting code was taken from an example found on matplotlib.org
