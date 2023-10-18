import os
import time
import numpy as np
import torch
try:
    from tensorboardX import SummaryWriter
except ImportError as error:
    print('tensorboard X not installed, visualizing wont be available')
    SummaryWriter = None


class Writer:
    def __init__(self, opt, dataset_descriptor=''):
        self.name = opt.name
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.log_name = os.path.join(self.save_dir, 'loss_log.txt')
        self.test_log = os.path.join(self.save_dir, 'test_log.txt')
        self.start_logs(dataset_descriptor)
        self.nexamples = 0
        self.ncorrect = 0
        if opt.is_train and not opt.no_vis and SummaryWriter is not None:
            self.display = SummaryWriter(
                logdir=os.path.join(self.opt.checkpoints_dir, self.opt.name) +
                "/tensorboard")  #comment=opt.name)
        else:
            self.display = None

    def start_logs(self, dataset_descriptor):
        """ creates test / train log files """
        if self.opt.is_train:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write(
                    f'================ Training Loss ({now}) ================\n')
                    
        else:
            with open(self.test_log, "a") as log_file:
                now = time.strftime("%c")
                log_file.write(
                    f'================ Testing on {dataset_descriptor} ({now}) ================\n')

    def print_current_losses(self,
                             epoch,
                             i,
                             losses,
                             t,
                             loss_type="total_loss"):
        """ prints train loss to terminal / file """
        if type(losses) == list:
            message = '(epoch: %d, iters: %d, time: %.3f)' \
                    % (epoch, i, t)
            for (loss_type, loss_value) in zip(loss_type, losses):
                message += ' %s: %.3f' % (loss_type, loss_value.item())
        else:
            message = '(epoch: %d, iters: %d, time: %.3f) loss: %.3f ' \
                    % (epoch, i, t, losses.item())
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def plot_loss(self, losses, epoch, i, n, loss_types):
        iters = i + (epoch - 1) * n
        if self.display:
            if type(losses) == list:
                for (loss_type, loss_value) in zip(loss_types, losses):
                    self.display.add_scalar('Loss/Train/' + loss_type,
                                            loss_value, iters)
            else:
                self.display.add_scalar('Loss/Train/', losses, iters)

    def plot_model_wts(self, model, epoch):
        if self.opt.is_train and self.display:
            for name, param in model.net.named_parameters():
                self.display.add_histogram(name,
                                           param.clone().cpu().data.numpy(),
                                           epoch)

    def print_acc(self, epoch, acc):
        """ prints test accuracy to terminal / file """
        message = 'epoch: {}, TEST ACC: [{:.4} %]' \
            .format(epoch, acc * 100)
        print(message)
        with open(self.test_log, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_classifier_metrics(self, epoch, acc, tp_rate, fp_rate, tn_rate, fn_rate):
        """ prints test classification metrics to terminal / file """
        message = 'epoch: {}, Accuracy: [{:.4} %] || TP: [{:.4} %] || FP: [{:.4} %] || TN: [{:.4} %] || FN: [{:.4} %] '\
            .format(epoch, acc * 100, tp_rate*100, fp_rate*100, tn_rate*100, fn_rate*100)
        print(message)
        with open(self.test_log, "a") as log_file:
            log_file.write('%s\n' % message)


    def print_reg(self, epoch, reg_stats):
        message = 'epoch: {} Translation error: [{:.4} +- {:.4} m] || Rotation error: [{:.4} +- {:.4} rad]\n' \
            .format(epoch, reg_stats['t_mean'], reg_stats['t_std'], reg_stats['r_mean'], reg_stats['r_std'])
        print(message)
        with open(self.test_log, "a") as log_file:
            log_file.write('%s\n' % message)
    

    def plot_class_stats(self, epoch, train_class_stats, test_class_stats=None):
        if self.display:
            #Accuracy
            if test_class_stats is not None:
                self.display.add_scalar('Classifier/Accuracy_Test', test_class_stats['accuracy'],
                                        epoch)
            self.display.add_scalar('Classifier/Accuracy_Train', train_class_stats['accuracy'],
                                    epoch)
            #Precision
            if test_class_stats is not None:
                if (test_class_stats['TP']+test_class_stats['FP'])>0:
                    test_precision = test_class_stats['TP']/(test_class_stats['TP']+test_class_stats['FP'])
                else:
                    test_precision = 0
                self.display.add_scalar('Classifier/Precision_Test', test_precision, epoch)
            
            if (train_class_stats['TP']+train_class_stats['FP'])>0:
                train_precision = train_class_stats['TP']/(train_class_stats['TP']+train_class_stats['FP'])
            else:
                train_precision = 0    
            self.display.add_scalar('Classifier/Precision_Train', train_precision, epoch)

            #Recall
            if test_class_stats is not None:
                if (test_class_stats['TP']+test_class_stats['FN'])>0:
                    test_recall = test_class_stats['TP']/(test_class_stats['TP']+test_class_stats['FN'])
                else:
                    test_recall = 0
                self.display.add_scalar('Classifier/Recall_Test', test_recall, epoch)

            if (train_class_stats['TP']+train_class_stats['FN'])>0:
                train_recall = train_class_stats['TP']/(train_class_stats['TP']+train_class_stats['FN'])
            else:
                train_recall = 0
            self.display.add_scalar('Classifier/Recall_Train', train_recall, epoch)

            #F1 Score
            if test_class_stats is not None:
                test_f1 = (2.0*test_precision*test_recall)/(test_precision+test_recall)
                self.display.add_scalar('Classifier/F1_Score_Test', test_f1, epoch)
            
            train_f1 = (2.0*train_precision*train_recall)/(train_precision+train_recall)
            self.display.add_scalar('Classifier/F1_Score_Train', train_f1, epoch)
            

    def plot_reg_stats(self, train_reg_stats, test_reg_stats, epoch):
        if self.display:
            self.display.add_scalar('Regression/Trans_mean_Train', train_reg_stats['t_mean'], epoch)
            self.display.add_scalar('Regression/Trans_mean_Test', test_reg_stats['t_mean'], epoch)
            self.display.add_scalar('Regression/Trans_std_Train', train_reg_stats['t_std'], epoch)
            self.display.add_scalar('Regression/Trans_std_Test', test_reg_stats['t_std'], epoch)

            self.display.add_scalar('Regression/Rot_mean_Train', train_reg_stats['r_mean'], epoch)
            self.display.add_scalar('Regression/Rot_mean_Test', test_reg_stats['r_mean'], epoch)
            self.display.add_scalar('Regression/Rot_std_Train', train_reg_stats['r_std'], epoch)
            self.display.add_scalar('Regression/Rot_std_Test', test_reg_stats['r_std'], epoch)


    def reset_counter(self):
        """
        counts # of correct examples
        """
        self.ncorrect = 0
        self.nexamples = 0
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def reset_reg_tracker(self):
        self.tr_error = None

    # def update_counter(self, ncorrect, nexamples):
    def update_counter(self, tp, fp, tn, fn, nexamples):
        self.nexamples += nexamples
        self.ncorrect += (tp+tn)
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def update_tr_error(self, tr_error):
        if self.tr_error is None:
            self.tr_error = tr_error
        else:
            self.tr_error = torch.cat([self.tr_error, tr_error], dim=0)

    @property
    def acc(self):
        return float(self.ncorrect) / self.nexamples
    
    @property
    def precision(self):
        return self.tp/(self.tp+self.fp)
    
    @property
    def recall(self):
        return self.tp/(self.tp+self.fn)
    
    @property
    def f1_score(self):
        return (2*self.precision*self.recall)/(self.precision+self.recall)
    
    @property
    def tp_rate(self):
        return float(self.tp/self.nexamples)
    
    @property
    def fp_rate(self):
        return float(self.fp/self.nexamples)
    
    @property
    def tn_rate(self):
        return float(self.tn/self.nexamples)
    
    @property
    def fn_rate(self):
        return float(self.fn/self.nexamples)
    

    @property
    def classification_stats(self):
        classifier_stats = {
            'accuracy': float(self.ncorrect) / self.nexamples,
            'TP': self.tp,
            'FP': self.fp,
            'TN': self.tn,
            'FN': self.fn,
        }
        return classifier_stats
    
    @property
    def regression_stats(self):
        """
        Returns a tuple (std, mean) where each element is a tensor
        of size 2 (first element corresponds to Translation, 
        second to Rotation)
        Translation mean = torch.std_mean(self.tr_error, dim=0)[1][0]
        """
        std_mean = torch.std_mean(self.tr_error, dim=0)
        regression_stats = {
            't_mean': std_mean[1][0],
            't_std': std_mean[0][0],
            'r_mean': std_mean[1][1],
            'r_std': std_mean[0][1]
        }
        return regression_stats


    def close(self):
        if self.display is not None:
            self.display.close()
