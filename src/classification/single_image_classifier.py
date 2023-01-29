import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import numpy as np
# import classification_model
# from classification_model import ImageClassificationModel


class ImageClassificationModel(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
            nn.Flatten(),
            nn.Linear(256 * 28 * 28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 5))

    def forward(self, xb):
        return self.network(xb)

    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def define_transform():
    transformer = torchvision.transforms.Compose(
        [  # Applying Augmentation
            torchvision.transforms.Resize((224, 224)),
            # torchvision.transforms.RandomHorizontalFlip(p=0.5),
            # torchvision.transforms.RandomVerticalFlip(p=0.5),
            # torchvision.transforms.RandomRotation(30),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return transformer


def imshow(img):
    """
        input: image in tensor format
        output: plt imshow
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def transform_image(img, tf):
    """
        input: image in .jpg format
        output: image in tensor format
    """
    img = tf(img)
    img_tensor = torch.unsqueeze(img, 0)
    return img_tensor


def load_model(path="../../classification_models/flower-cnn.pth"):
    """
        input: path to the model
        output: model
    """
    model = ImageClassificationModel()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model


def predict_image(img_tensor, model):
    """
        input: image in tensor format, model
        output: prediction
    """
    with torch.no_grad():
        model.eval()

        labels = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        output = model(img_tensor)
        _, prediction = torch.max(output, 1)
        percentage = torch.nn.functional.softmax(output, dim=1)[0] * 100
        imshow(img_tensor[0])
        print(labels[prediction[0]], percentage[prediction[0]].item())
        return prediction, percentage

