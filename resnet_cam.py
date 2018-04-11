import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ResNet_CAM(nn.Module):
    def __init__(self):
        super(ResNet_CAM, self).__init__()
        pretrained_model = torchvision.models.resnet50(pretrained=True)
        self.conv1 = pretrained_model.conv1
        self.bn1 = pretrained_model.bn1
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        self.layer4 = pretrained_model.layer4
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, 10)

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        fs = self.gap(c5)
        fs = fs.view(fs.size(0), -1)
        fs = self.fc(fs)
        return F.log_softmax(fs, dim=1), c5

if __name__ == '__main__':
    import torch
    from torch.autograd import Variable
    data = Variable(torch.randn([1, 3, 224, 224]))
    net = ResNet_CAM()
    p, fms = net(data)
    print(p.size(), fms.size())