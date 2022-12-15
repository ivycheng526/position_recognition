import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
import pickle as pkl
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from CNN import CNN_model
import sys
def print_loss(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))

    sys.stdout.flush()

np.random.seed(3)


class MyDataset(Dataset):
    def __init__(self, path, train=True):
        self.train = train
        path = path + ("train/" if train else "test/")
        self.pathX = path + 'X/'
        self.pathY = path + 'Y/'
        self.data = os.listdir(self.pathX)

    def __getitem__(self, idx):
        if self.train:
            f = self.data[idx]
            # X
            # rgb images
            angle_id = int(np.random.choice([0, 1, 2], 1))
            img = cv2.imread(self.pathX + f + "/rgb/{}.png".format(angle_id))
            # augmentation
            img = self.color_jitter(img)
            img = TF.to_tensor(img)
            img = img / 255
            # normalzie
            img = TF.normalize(img, [0.5], [0.5])

            # depth images
            depth = np.load(self.pathX + f + '/depth.npy')[angle_id]
            depth = TF.to_tensor(depth)
            depth /= 1000.0
            # normalzie
            depth = TF.normalize(depth, [0.5], [0.5])

            # read field ID
            field_id = pkl.load(open(self.pathX + f + '/field_id.pkl', 'rb'))

            # Y
            Y = np.load(self.pathY + f + '.npy')
            Y *= 1000.0
            Y = torch.FloatTensor(Y)

            return (img, depth, field_id), Y

        else:
            f = self.data[idx]
            # rgb images
            angle_id = int(np.random.choice([0, 1, 2], 1))
            img = cv2.imread(self.pathX + f + "/rgb/{}.png".format(angle_id))
            # #augmentation
            # img =self.color_jitter(img)
            img = Image.fromarray(img)
            img = TF.to_tensor(img)
            img = img / 255
            # normalzie
            img = TF.normalize(img, [0.5], [0.5])

            # depth images
            depth = np.load(self.pathX + f + '/depth.npy')[angle_id]
            depth = Image.fromarray(depth)
            depth = TF.to_tensor(depth)
            depth /= 1000.0
            # normalzie
            depth = TF.normalize(depth, [0.5], [0.5])

            # read field ID
            field_id = pkl.load(open(self.pathX + f + '/field_id.pkl', 'rb'))

            return (img, depth, field_id)

    def __len__(self):
        return len(self.data)

    # color_jitter the images randomly
    def color_jitter(self, image):
        image = Image.fromarray(image)
        color_jitter = transforms.ColorJitter(brightness=0.3,
                                              contrast=0.3,
                                              saturation=0.3,
                                              hue=0.1)
        image = color_jitter(image)
        return image


dataset = MyDataset('./lazydata/')
test_dataset = MyDataset('./lazydata/',train=False)

# split the dataset into validation and test sets
len_valid_set = int(0.1*len(dataset))
len_train_set = len(dataset) - len_valid_set
train_dataset , valid_dataset = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])

# shuffle and batch the datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=0)
print("The length of Train set is {}".format(len_train_set))
print("The length of Valid set is {}".format(len_valid_set))

# Do not shuffle the test and example images
test_loader =torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

import time

train_mode = True
continue_trained = False
if train_mode:
    model = CNN_model()
    model.cuda()

    if continue_trained:
        # loading the saved best model
        try:
            print("Loading model...")
            model.load_state_dict(torch.load('./best_model.pth'))
        except:
            raise ValueError("No trained model!")

    # model.cuda()# use gpu to speed up the training process
    criterion = nn.MSELoss()  # using Mean Square error as loss function
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # using adam optimizer ,set learning rate
    loss_min = np.inf
    num_epochs = 70  # set number of epochs

    start_time = time.time()

    train_loss = []
    val_loss = []
    for epoch in range(1, num_epochs + 1):
        loss_train = 0
        loss_valid = 0
        running_loss = 0
        model.train()

        for step in range(1, len(train_loader) + 1):
            (img, depth, field_id), Y = next(iter(train_loader))
            input = torch.cat((img, depth), 1).cuda()
            Y = Y.cuda()
            pred = model(input)
            # clear all the gradients before calculating them
            optimizer.zero_grad()

            # find the loss for the current step
            loss_train_step = criterion(pred, Y)

            # calculate the gradients
            loss_train_step.backward()

            # update the parameters
            optimizer.step()

            loss_train += loss_train_step.item()
            running_loss = loss_train / step

            print_loss(step, len(train_loader), running_loss, 'train')

        model.eval()
        with torch.no_grad():
            for step in range(1, len(valid_loader) + 1):
                (img, depth, field_id), Y = next(iter(valid_loader))
                input = torch.cat((img, depth), 1).cuda()
                Y = Y.cuda()
                pred = model(input)

                # find the loss for the current step
                loss_valid_step = criterion(pred, Y)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid / step

                print_loss(step, len(valid_loader), running_loss, 'valid')

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)
        # loss_valid = np.sqrt(loss_valid)

        print('*************************************************')
        print('Epoch:{} finished!   Train Loss: {:.5f}  Valid Loss: {:.5f}'.format(epoch, loss_train, loss_valid))
        print('*************************************************')
        train_loss.append(loss_train)
        val_loss.append(loss_valid)

        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(model.state_dict(), './best_model.pth')
            print("Minimum Validation Loss of {:.5f} at epoch {}/{}".format(loss_min, epoch, num_epochs))
            print('Model Saved Successfully! ')
            print('*************************************************')

    print('Training Finished! ')
    print("Total Running Time : {} s".format(time.time() - start_time))

    np.save('./train_loss',np.array(train_loss))
    np.save('./val_loss',np.array(val_loss))

    plt.figure()
    plt.plot(train_loss,label = "train loss")
    plt.plot(val_loss, label = 'validation loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Loss during training")
    plt.savefig('./loss.png')
    plt.show()
else:
    outfile = 'submission.csv'
    output_file = open(outfile, 'w')
    titles = ['ID', 'FINGER_POS_1', 'FINGER_POS_2', 'FINGER_POS_3', 'FINGER_POS_4', 'FINGER_POS_5', 'FINGER_POS_6',
              'FINGER_POS_7', 'FINGER_POS_8', 'FINGER_POS_9', 'FINGER_POS_10', 'FINGER_POS_11', 'FINGER_POS_12']
    preds = []
    file_ids = []
    model = CNN_model()
    model.cuda()

    # loading the saved best model
    try:
        print("Loading model...")
        model.load_state_dict(torch.load('./best_model_70e.pth'))
    except:
        raise ValueError("No trained model!")

    model.eval()

    for step, test_data in enumerate(test_loader):
        (test_img, test_depth, field_id) = test_data
        input = torch.cat((test_img, test_depth), 1).cuda()
        test_pred = model(input)
        test_pred = test_pred/1000.0
        preds.append(test_pred[0].cpu().detach().numpy())
        file_ids.append(field_id)

    df = pd.concat([pd.DataFrame(file_ids), pd.DataFrame.from_records(preds)], axis=1, names=titles)
    df.columns = titles
    df.to_csv(outfile, index=False)
    print("Written to csv file {}".format(outfile))




