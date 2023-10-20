import torch
from . import networks
from os.path import join


class GraspEvalModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> sampling / evaluation)
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.loss = None
        self.pcs = None
        self.pc_features = None
        # load/define networks
        self.net = networks.define_network(opt, 
                                           opt.init_type,
                                           opt.init_gain,
                                           self.device)
        self.criterion = networks.define_loss(opt)
        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(),
                                              lr=opt.lr,
                                              betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch, self.is_train)

    def set_input(self, data):
        ######## PointnetSAModule input and output #####################
        """        
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """
        input_pcs = data['pcd'][:,:,:3].contiguous()
        input_pc_features = torch.transpose(data['pcd'],1,2).contiguous()  #See above on why transpose is needed
        self.label_targets = data['label'].float().contiguous().to(self.device)
        #self.tr_targets = data['tr'].float().contiguous().to(self.device)
        self.pcs = input_pcs.to(self.device).requires_grad_(self.is_train)
        self.pc_features = input_pc_features.to(self.device).requires_grad_(self.is_train)

    def forward(self):
        return self.net(self.pcs, self.pc_features, train=self.is_train)

    def backward(self, out):
        # grasp_classification, tr_regression = out
        grasp_classification = out
        # self.classification_loss, self.regression_loss = self.criterion(
        self.classification_loss = self.criterion(
            grasp_classification.squeeze(),
            self.label_targets,
            device=self.device,
            use_pos_weight=True,
            #tr_regression.squeeze(),
            #self.tr_targets,
        )
        # if torch.isnan(self.regression_loss):
        #     self.loss = self.classification_loss
        # else:
        #     self.loss = self.classification_loss + self.regression_loss
        self.loss = self.classification_loss
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()


    def load_network(self, which_epoch, train=True):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('Loading the model from %s' % load_path)
        checkpoint = torch.load(load_path, map_location=self.device)
        if hasattr(checkpoint['model_state_dict'], '_metadata'):
            del checkpoint['model_state_dict']._metadata
        net.load_state_dict(checkpoint['model_state_dict'])
        if train:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.opt.epoch_count = checkpoint["epoch"]
        else:
            net.eval()

    def save_network(self, net_name, epoch_num):
        """save model to disk"""
        save_filename = '%s_net.pth' % (net_name)
        save_path = join(self.save_dir, save_filename)
        torch.save(
            {
                'epoch': epoch_num + 1,
                'model_state_dict': self.net.module.cpu().state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }, save_path)

        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.net.cuda(self.gpu_ids[0])

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.12f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            # grasp_classification, tr_regression = self.forward()
            grasp_classification = self.forward()
            predicted = torch.round(torch.sigmoid(grasp_classification)).squeeze()
            # True Positives (TP): Number of samples correctly predicted as “positive.”
            true_pos = torch.logical_and(   
                torch.gt(self.label_targets,0),
                torch.gt(predicted,0)).sum().item()
            # False Positives (FP): Number of samples wrongly predicted as “positive.”
            false_pos = torch.logical_and(
                ~torch.gt(self.label_targets,0),
                torch.gt(predicted,0)).sum().item()
            # True Negatives (TN): Number of samples correctly predicted as “negative.”
            true_neg = torch.logical_and(
                ~torch.gt(self.label_targets,0),
                ~torch.gt(predicted,0)).sum().item()
            # False Negatives (FN): Number of samples wrongly predicted as “negative.”
            false_neg = torch.logical_and(
                torch.gt(self.label_targets,0),
                ~torch.gt(predicted,0)).sum().item()
            # Compute error in regression          
            #tr_error = self.tr_targets - tr_regression
            return true_pos, false_pos, true_neg, false_neg, len(self.label_targets)#, tr_error
