import os
import time

import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
from torch.nn import DataParallel, NLLLoss
from torch.utils.data import DataLoader

import pytorch_utils
from config import Config
from data_generator import DataGenerator
from lr_scheduler import LRScheduler
from resnet_cam import ResNet_CAM

root_path = '/home/storage/lsy/interpret_cnn/'

def print_log(epoch, lr, train_metrics, train_time, val_metrics, accuracy, val_time, save_dir, log_mode):
    train_metrics = np.mean(train_metrics, axis=0)
    val_metrics = np.mean(val_metrics, axis=0)
    str0 = 'Epoch %03d (lr %.7f)' % (epoch, lr)
    str1 = 'Train:      time %3.2f loss: %2.4f' % (train_time, train_metrics)
    str2 = 'Validation: time %3.2f loss: %2.4f accuracy: %2.4f' % (val_time, val_metrics, accuracy)

    print(str0)
    print(str1)
    print(str2 + '\n')
    if epoch > 1:
        log_mode = 'a'
    f = open(save_dir + 'kpt_train_log.txt', log_mode)
    f.write(str0 + '\n')
    f.write(str1 + '\n')
    f.write(str2 + '\n\n')
    f.close()

def train(data_loader, net, loss, optimizer, lr):
    start_time = time.time()

    net.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    for i, (imgs, lbls) in enumerate(data_loader):
        imgs = Variable(imgs.cuda(async=True))
        lbls = Variable(lbls.cuda(async=True)).squeeze()
        outputs, _ = net(imgs)
        loss_output = loss(outputs, lbls)
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()
        metrics.append(loss_output.data[0])
    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    return metrics, end_time - start_time

def validate(data_loader, net, loss):
    start_time = time.time()
    net.eval()
    metrics = []
    correct = 0
    total = 0
    for i, (imgs, lbls) in enumerate(data_loader):
        imgs = Variable(imgs.cuda(async=True), volatile=True)
        lbls = Variable(lbls.cuda(async=True), volatile=True).squeeze()
        outputs, _ = net(imgs)
        loss_output = loss(outputs, lbls)
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(lbls.data.view_as(pred)).long().cpu().sum()
        total += lbls.size(0)
        metrics.append(loss_output.data[0])
    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    return metrics, float(correct) / total, end_time - start_time


if __name__ == '__main__':
    batch_size = 64
    workers = 32
    n_gpu = pytorch_utils.setgpu('5,6')
    epochs = 100
    base_lr = 1e-3
    save_dir = root_path + 'checkpoints/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    resume = False

    config = Config()
    net = ResNet_CAM()

    start_epoch = 1
    lr = base_lr
    best_val_loss = float('inf')
    log_mode = 'w'
    if resume:
        checkpoint = torch.load(save_dir + '008.ckpt')
        start_epoch = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        best_val_loss = checkpoint['best_val_loss']
        net.load_state_dict(checkpoint['state_dict'])
        log_mode = 'a'

    loss = NLLLoss().cuda()
    net = net.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)

    train_dataset = DataGenerator(config, phase='train')
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=workers,
                              pin_memory=True)
    val_dataset = DataGenerator(config, phase='val')
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=workers,
                            pin_memory=True)
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=1e-4)
    lrs = LRScheduler(lr, epochs, patience=3, factor=0.1, min_lr=1e-5, early_stop=5, best_loss=best_val_loss)
    for epoch in range(start_epoch, epochs + 1):
        train_metrics, train_time = train(train_loader, net, loss, optimizer, lr)
        val_metrics, accuracy, val_time = validate(val_loader, net, loss)

        print_log(epoch, lr, train_metrics, train_time, val_metrics, accuracy, val_time, save_dir, log_mode)

        val_loss = np.mean(val_metrics)
        lr = lrs.update_by_rule(val_loss)
        if val_loss < best_val_loss or epoch%10 == 0 or lr is None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            state_dict = net.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'lr': lr,
                'best_val_loss': best_val_loss},
                os.path.join(save_dir, '%03d.ckpt' % epoch))

        if lr is None:
            print('Training is early-stopped')
            break


