import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
import torch
from tqdm import tqdm

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def model_summary(model, input_size):
    summary(model, input_size=input_size)

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def display_image(image, label):
    plt.imshow(image.squeeze(), cmap='gray')
    print('label:', label)
    plt.waitforbuttonpress()

def display_image_grid(images, labels):
    grid = torchvision.utils.make_grid(images, nrow=10)
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    print('labels:', labels)
    plt.waitforbuttonpress()

def visualize_Data(train_loader):
    batch_data, batch_label = next(iter(train_loader))

    fig = plt.figure()

    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(batch_data[i].squeeze(0), cmap='gray')
        plt.title(batch_label[i].item())
        plt.xticks([])
        plt.yticks([])
    plt.waitforbuttonpress()


class NNPipeLine:
    def __init__(self,model, device):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

        self.model = model
        self.device = device

    def train(self, train_loader, optimizer, criterion):

        self.model.train()
        pbar = tqdm(train_loader)

        # Data to plot accuracy and loss graphs
        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            # Predict
            pred = self.model(data)         # Forward pass
            loss = criterion(pred, target)  # Calculate loss
            loss.backward()                 # Backpropagation
            optimizer.step()                # Update weights
            train_loss += loss.item()       # Accumulate loss

            correct += get_num_correct(pred, target)    # Number of correct predictions
            processed += len(data)                      # Number of samples processed.

            pbar.set_description(
                desc=f'Train: Loss={loss.item(): 0.4f} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')

        self.train_acc.append(100 * correct / processed)
        self.train_losses.append(train_loss / len(train_loader))

    def test(self, test_loader, criterion):
        self.model.eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += criterion(output, target).item()  # sum up batch loss

                correct += get_num_correct(output, target)

        test_loss /= len(test_loader.dataset)
        self.test_acc.append(100. * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    def print_performance(self):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")
        plt.waitforbuttonpress()



