import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from .losses import classification_BCE_with_logits
import pointnet2_ops.pointnet2_modules as pointnet2


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(
                0, epoch + 1 + 1 - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
        elif classname.find('BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type, init_gain, device):
    if torch.cuda.is_available():
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        net = torch.nn.DataParallel(net)
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    net.to(device)
    return net


def define_network(opt, init_type, init_gain, device):
    net = GraspEvaluatorNetwork(opt.model_scale, opt.pointnet_radius,
                             opt.pointnet_nclusters, device)
    return init_net(net, init_type, init_gain, device)


def define_loss(opt):
    return classification_BCE_with_logits


def base_network(pointnet_radius, pointnet_nclusters, scale, in_features):
    sa1_module = pointnet2.PointnetSAModule(
        npoint=pointnet_nclusters,
        radius=pointnet_radius,
        nsample=64,
        mlp=[in_features, 2**(3+scale), 2**(3+scale), 2**(4+scale)])  #64,64,128 for scale=3
    sa2_module = pointnet2.PointnetSAModule(
        npoint=32,
        radius=0.04,
        nsample=128,
        mlp=[2**(4+scale), 2**(4+scale), 2**(4+scale), 2**(5+scale)]) #128,128,128,256 for scale=3

    sa3_module = pointnet2.PointnetSAModule(
        mlp=[2**(5+scale), 2**(5+scale), 2**(5+scale), 2**(6+scale)]) #256,256,256,512 for scale=3

    sa_modules = nn.ModuleList([sa1_module, sa2_module, sa3_module])
    fc_layer = nn.Sequential(nn.Linear(2**(6+scale), 2**(7+scale)),
                             nn.BatchNorm1d(2**(7+scale)), nn.ReLU(True),
                             nn.Linear(2**(7+scale), 2**(7+scale)),
                             nn.BatchNorm1d(2**(7+scale)), nn.ReLU(True))
    return nn.ModuleList([sa_modules, fc_layer])


class GraspEvaluatorNetwork(nn.Module):
    def __init__(self,
                 model_scale=3,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128,
                 device="cpu"):
        super(GraspEvaluatorNetwork, self).__init__()
        self.create_evaluator(pointnet_radius, model_scale, pointnet_nclusters)
        self.device = device

    def create_evaluator(self, pointnet_radius, model_scale,
                         pointnet_nclusters):
        # The number of input features for the evaluator is 4: the x, y, z
        # position of the concatenated gripper and object point-clouds and an
        # extra binary feature, which is 0 for the object and 1 for the gripper,
        # to tell these point-clouds apart
        self.evaluator = base_network(pointnet_radius, pointnet_nclusters,
                                      model_scale, 4)
        self.predictions_logits = nn.Linear(2**(7+model_scale), 1)
        # self.tr_regression = nn.Sequential(nn.Linear(2**(7+model_scale), 2**(5+model_scale)),
        #                                    nn.BatchNorm1d(2**(5+model_scale)), nn.ReLU(True),
        #                                    nn.Linear(2**(5+model_scale), 2**(3+model_scale)),
        #                                    nn.BatchNorm1d(2**(3+model_scale)), nn.ReLU(True),
        #                                    nn.Linear(2**(3+model_scale), 2), nn.ReLU(True))
                                           

    def evaluate(self, xyz, xyz_features):
        for module in self.evaluator[0]:
            xyz, xyz_features = module(xyz, xyz_features)
        return self.evaluator[1](xyz_features.squeeze(-1))

    def forward(self, pc, pc_features, train=True):
        x = self.evaluate(pc, pc_features)
        return self.predictions_logits(x)#, self.tr_regression(x)


