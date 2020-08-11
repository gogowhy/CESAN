import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)] 
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, freeze=False, dropout=0,bn=False,feature_num=0, max_pool=False ):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        if freeze:
            for p in self.parameters():
                p.requires_grad=False
        self.classifier = ClassBlock(2048, class_num)
        self.num_features = feature_num
        # self.has_embedding = feature_num >0
        self.dropout = dropout
        self.bn = bn
        self.max_pool = max_pool
        if self.max_pool:
            self.pool = nn.AdaptiveMaxPool2d((1,1))

        # if self.has_embedding:
        #     self.feat = nn.Linear(2048, self.num_features)
        #     self.feat_bn = nn.BatchNorm1d(self.num_features)
        #     init.kaiming_normal(self.feat.weight, mode='fan_out')
        #     init.constant(self.feat.bias, 0)
        #     init.constant(self.feat_bn.weight, 1)
        #     init.constant(self.feat_bn.bias, 0)
        # else:
            # Change the num_features to CNN output channels
            self.num_features = 2048
        if self.dropout>0:
            self.drop = nn.Dropout(self.dropout)

        if self.bn:
            self.feat_bn2d = nn.BatchNorm2d(2048) #may not be used, not working on caffe
            init.constant(self.feat_bn2d.weight,1) #initialize BN, may not be used
            init.constant(self.feat_bn2d.bias,0)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.max_pool:
            x = self.pool(x)
        else:
            x = self.model.avgpool(x)

        x = torch.squeeze(x)
        # if self.has_embedding:
        #     x = self.feat(x)
        #     x = self.feat_bn(x)
        if self.bn:
            x = F.normalize(x)
        # print('x:', x.shape)
        # elif self.has_embedding:
        #     x = F.relu(x)
        # if self.dropout > 0:
        #     x = self.drop(x)
        # x = self.classifier(x)
        return x
