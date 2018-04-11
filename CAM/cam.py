import numpy as np
import torch
from torch.autograd import Variable
from torch.backends import cudnn
import cv2
from torch.nn import DataParallel, NLLLoss
from torch.utils.data import DataLoader

from data_generator import DataGenerator
import pytorch_utils
from config import Config
from resnet_cam import ResNet_CAM
import mnist

root_path = '/home/storage/lsy/interpret_cnn/'

def draw_heatmap(image, heatmap):
    alpha = 0.5
    heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap_img, alpha, image, 1 - alpha, 0)
    return fin

if __name__ == '__main__':
    config = Config()
    n_gpu = pytorch_utils.setgpu('6')
    net = ResNet_CAM()
    checkpoint = torch.load(root_path + 'checkpoints/020.ckpt')  # must before cuda
    net.load_state_dict(checkpoint['state_dict'])
    net = net.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)
    net.eval()


    imgs, lbls = mnist.read(dataset='test', path=config.mnist_dir)
    idcs = np.random.randint(0, len(lbls), size=(3, ))
    img_pad = np.zeros([3, config.img_size, config.img_size], dtype=np.float32)
    classes = []
    for idx in idcs:
        img = imgs[idx].astype(np.float32)
        ih, iw = img.shape
        img = (img / 255. - config.mnist_mean) / config.mnist_std
        x, y = np.random.randint(0, config.img_size - ih, (2,), dtype=np.int16)
        for k in range(3):
            img_pad[k, y:y + ih, x:x + iw] = img
        classes.append(lbls[idx])
    data = np.expand_dims(img_pad, 0).copy()
    data = torch.from_numpy(data)

    data = Variable(data.cuda(async=True), volatile=True)
    output, fm = net(data)

    probs, index = output.data.squeeze().sort(0, True)
    print(classes)
    print(index.cpu().numpy())

    w = list(net.parameters())[-2]
    fm = fm[0]
    M = torch.mm(w, fm.view(fm.size(0), -1))
    M = M.view(M.size(0), fm.size(1), fm.size(2)).data.cpu().numpy()
    M = (M - np.amin(M)) / (np.amax(M) - np.amin(M))
    img_ori = np.uint8((img_pad * config.mnist_std + config.mnist_mean) * 255).transpose(1, 2, 0)
    for cls, mm in enumerate(M):
        mm = np.uint8(255 * mm)
        mm = cv2.resize(mm, (config.img_size, config.img_size))
        draw = draw_heatmap(img_ori, mm)
        cv2.imwrite(root_path + '/tmp/%d.png' % cls, draw)