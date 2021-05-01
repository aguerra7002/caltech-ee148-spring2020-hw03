from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

import os


def euclidean_boys(features, data, target):
    big_ol_grid = np.zeros((28 * 4, 28 * 9))
    for i in range(4):
        print(i)
        to_comp = features[i]
        # Fill in the left most features
        big_ol_grid[28*i:28*(i+1),0:28] = data[i]
        stats = []
        for j in range(4, 10000):
            dist = np.linalg.norm(to_comp - features[j])
            stats.append((j, dist))
        stats_sorted = sorted(stats, key=lambda tup: tup[1])
        for j in range(8):
            big_ol_grid[28*i:28*(i+1), 28*(j+1):28*(j+2)] = data[stats_sorted[j][0]]
    plt.imshow(big_ol_grid)
    plt.show()



class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # Turns it into a feature vector (i.e., all the layers except the last one)
    def to_feature(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()   # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()               # Clear the gradient
        output = model(data)                # Make predictions
        loss = F.nll_loss(output, target)   # Compute loss
        loss.backward()                     # Gradient computation
        optimizer.step()                    # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()    # Set the model to inference mode
    test_loss = 0
    correct = 0
    test_num = 0
    incorrect = 0
    confusion_mat = torch.zeros((10,10))
    with torch.no_grad():   # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # Keeps a confusion matrix handy
            confusion_mat += np.array(confusion_matrix(target.cpu(), pred.cpu()))
            correct_inds = pred.eq(target.view_as(pred))
            # The following code plots the first 9 incorrectly classified images in the test set.
            # if incorrect < 9:
            #     for i in range(len(correct_inds)):
            #         if correct_inds[i][0].item() == False:
            #             plt.imshow(data[i][0].cpu())
            #             plt.savefig(str(incorrect) + ".jpg")
            #             incorrect += 1
            correct += correct_inds.sum().item()
            test_num += len(data)
            # Get the TSNE plot for the data
            features = model.to_feature(data)
            # to_plot = TSNE(n_components=2).fit_transform(features.cpu())
            # # Make the TSNE Plot:
            # colors = cm.rainbow(np.linspace(0, 1, 10))
            # for digit in range(10):
            #     inds = [x for x in range(target.shape[0]) if target[x] == digit]
            #     plt.scatter(to_plot[inds, 0], to_plot[inds, 1], color=colors[digit], label=str(digit))
            # plt.legend()
            # plt.title("TSNE plot for feature vectors")
            # plt.show()
            euclidean_boys(features.cpu(), data.cpu(), target.cpu())


    test_loss /= test_num




    # Plots the kernels
    # for name, param in model.named_parameters():
    #     if name == "conv1.weight":
    #         print(param.shape)
    #         for i in range(9):
    #             plt.imshow(param[i,0].detach().cpu())
    #             plt.savefig("ker" + str(i) + ".jpg")


    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, test_num,
        100. * correct / test_num))


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = ConvNet().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=False,
                transform=transforms.Compose([       # Data preprocessing
                    transforms.ToTensor(),           # Add data augmentation here
                    transforms.Normalize((0.1307,), (0.3081,))
                    # transforms.GaussianBlur(3),
                    # transforms.RandomAutocontrast()
                ]))

    # Determines how much of the train data to actually use
    proportion = 1/16

    subset_indices_train = []
    subset_indices_valid = []
    for k in range(10):
        k_inds = [x for x in range(len(train_dataset)) if train_dataset[x][1] == k]
        random_inds =list(SubsetRandomSampler(k_inds))
        pivot1 = int(.85 * proportion* len(random_inds))
        pivot2 = int(.85 * len(random_inds))
        subset_indices_train += random_inds[:pivot1]
        subset_indices_valid += random_inds[pivot2:]


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=SubsetRandomSampler(subset_indices_train)
    )
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.test_batch_size,
        sampler=SubsetRandomSampler(subset_indices_valid)
    )

    # Load your model [fcNet, ConvNet, Net]
    model = ConvNet().to(device)

    # Try different optimzers here [Adam, SGD, RMSprop]
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Set your learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, train_loader)
        test(model, device, val_loader)
        scheduler.step()    # learning rate scheduler

        # You may optionally save your model at each epoch here

    if args.save_model:
        torch.save(model.state_dict(), "mnist_model_" + str(proportion) + ".pt")



if __name__ == '__main__':
    main()
    # Here's where we plot
    # These are the results for accuracy we got.
    # train_accuracy = [50343/50995, 25080/25495, 12507/12745, 6231/6370, 3093/3182]
    # test_accuracy = [.9864, .9798, .9751, .9641, 0.9587]
    # x_axis = np.array([50995, 25495, 12745, 6370, 3182])
    # plt.plot(np.log(x_axis), np.log(np.array(train_accuracy)), label="Train Accuracy")
    # plt.plot(np.log(x_axis), np.log(np.array(test_accuracy)), label="Test Accuracy")
    # plt.legend()
    # plt.xlabel("Log(Training Samples)")
    # plt.ylabel("Log(Accuracy)")
    # plt.title("Train/Test Accuracies vs. Training Size")
    # plt.show()