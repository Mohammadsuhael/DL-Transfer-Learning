# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
develop an image classification model using transfer learning with VGG19 architecture for the given dataset.
## Neural Network Model
<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/43600277-359e-457b-9767-e954c59aac57" />

## DESIGN STEPS
### STEP 1: 
Collect and organize your images into folders (one folder per class).<br>
Resize all images to 224 × 224 (because VGG19 expects this size).<br>

### STEP 2: 
Use a library like TensorFlow or Keras.<br>
Normalize pixel values (scale between 0 and 1).<br>
Apply data augmentation (rotation, flip, zoom) to improve accuracy.<br>

### STEP 3: 
Load VGG19 with pre-trained ImageNet weights.<br>
Remove the top (final classification layer).<br>
Freeze the base layers so their weights do not change during training.<br>

### STEP 4: 
#### Add:
Flatten layer<br>
Dense (fully connected) layer<br>
Dropout layer (to reduce overfitting)<br>
Final Dense layer with Softmax (number of neurons = number of classes)<br>

### STEP 5: 
#### Choose:
Optimizer (e.g., Adam)<br>
Loss function (e.g., categorical crossentropy)<br>
Metrics (accuracy)<br>
Train the model using training data.<br>
Validate using validation data.<br>

### STEP 6: 
Test the model on unseen data.<br>
If accuracy is low:<br>
Unfreeze some top VGG19 layers.<br>
Train again with a low learning rate (fine-tuning).<br>
Save the final trained model.<br>
## PROGRAM

### Name: Mohammad Suhael

### Register Number:212224230164

```python

# Load Pretrained Model and Modify for Transfer Learning
model = models.vgg19(weights=VGG19_Weights.DEFAULT)

# Modify the final fully connected layer to match the dataset classes

model.classifier[-1] = nn.Linear(model.classifier[-1].in_features,1)

# Include the Loss function and optimizer

criterion =nn.BCEWithLogitsLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)

# Train the model

def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses=[]
    val_losses=[]
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))
        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
          for images,labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            val_loss += loss.item()

        val_losses.append(val_loss/len(test_loader))
        model.train()
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: Mohammad Suhael")
    print("Register Number: 212224230164")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()



```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="805" height="654" alt="image" src="https://github.com/user-attachments/assets/94b089b2-4b4a-425e-b812-5a0d1fba68fe" />

## Confusion Matrix

<img width="753" height="681" alt="image" src="https://github.com/user-attachments/assets/62510e99-e04e-4c9f-965d-9925b1bc29b9" />

## Classification Report
<img width="575" height="252" alt="image" src="https://github.com/user-attachments/assets/8298f349-41d3-4d52-8d4c-c4aaa7bb8813" />

### New Sample Data Prediction
<img width="482" height="461" alt="image" src="https://github.com/user-attachments/assets/ef253050-f79e-408e-913c-2e46e1566acd" />
<img width="479" height="455" alt="image" src="https://github.com/user-attachments/assets/54a2ec0a-f897-4098-8c37-9b8ccdcebc33" />


## RESULT
Thus, the image classification model using transfer learning with VGG19 architecture for the given dataset has been executed successfully.
