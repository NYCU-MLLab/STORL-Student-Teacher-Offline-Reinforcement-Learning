import os
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from convlab2.util.train_util import to_device
import math


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLE_Trainer_Abstract():
    def __init__(self, manager, cfg):
        self._init_data(manager, cfg)
        self.policy = None
        self.policy_optim = None
        self.policy_expert = None
        self.gamma_pos = 2
        self.gamma_neg = 2
    def _init_data(self, manager, cfg):
        self.data_train = manager.create_dataset('train', cfg['batchsz'])
        self.data_valid = manager.create_dataset('val', cfg['batchsz'])
        self.data_test = manager.create_dataset('test', cfg['batchsz'])
        self.save_dir = cfg['save_dir']
        self.print_per_batch = cfg['print_per_batch']
        self.save_per_epoch = cfg['save_per_epoch']
        self.multi_entropy_loss = nn.MultiLabelSoftMarginLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=0.2, p=2)

    def asl_loss(self,a,target_a):
        x_sigmoid = torch.sigmoid(a)
        xs_pos = x_sigmoid
        xs_neg = 1-x_sigmoid
        # xs_neg = (xs_neg + 0.05).clamp(max=1)

        los_pos = target_a * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - target_a) * torch.log(xs_neg.clamp(min=1e-8))
        loss1 = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            # if self.disable_torch_grad_focal_loss:
            #     torch.set_grad_enabled(False)
            pt0 = xs_pos * target_a
            pt1 = xs_neg * (1 - target_a)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * target_a + self.gamma_neg * (1 - target_a)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            # if self.disable_torch_grad_focal_loss:
            #     torch.set_grad_enabled(True)
            loss1 *= one_sided_w
        return -loss1.sum(dim=-1).mean()

    def policy_loop(self, data):
        s, target_a, s_pos, s_neg1, s_neg2 = to_device(data)
        
        a,z= self.policy(s)

        a1,positives = self.policy(s_pos)

        a2,negatives = self.policy(s_neg1)

        loss_act = self.multi_entropy_loss(a, target_a)

        loss_tot = 0.

        loss_act = loss_act.mean(dim=-1)
        return loss_act.mean(),loss_tot
        
    def imitating(self, epoch):
        """
        pretrain the policy by simple imitation learning (behavioral cloning)
        """
        self.policy.train()
        loss_act = 0.
        for i, data in enumerate(self.data_train):
            self.policy_optim.zero_grad()
            los_a,los_b = self.policy_loop(data)
            loss_act+= los_a.item()
            losses = los_a+los_b
            losses.backward()
            self.policy_optim.step()
            
            if (i+1) % self.print_per_batch == 0:
                loss_act /= self.print_per_batch
                logging.debug('<<dialog policy>> epoch {}, iter {}, loss_tot:{}'.format(epoch, i, loss_act))
                loss_act = 0.
        
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
        self.policy.eval()
    
    def imit_test(self, epoch, best):
        """
        provide an unbiased evaluation of the policy fit on the training dataset
        """
        loss_act = 0.
        for i, data in enumerate(self.data_valid):
            los_a,los_b = self.policy_loop(data)
            loss_act+= los_a.item()

        loss_act /= len(self.data_valid)

        logging.debug('<<dialog policy>>validation, epoch {}, iter {}, loss_tot:{}'.format(epoch, i,loss_act))
        if loss_act < best:
            logging.info('<<dialog policy>> best model saved')
            best = loss_act
            self.save(self.save_dir, 'best')
            
        loss_act = 0.

        for i, data in enumerate(self.data_test):
            los_a,los_b = self.policy_loop(data)
            loss_act+= los_a.item()

        loss_act /= len(self.data_test)

        logging.debug('<<dialog policy>>validation, epoch {}, iter {}, loss_tot:{}'.format(epoch, i,loss_act))
        return best

    def test(self):
        def f1(a, target):
            TP, FP, FN = 0, 0, 0
            real = target.nonzero().tolist()
            predict = a.nonzero().tolist()
            for item in real:
                if item in predict:
                    TP += 1
                else:
                    FN += 1
            for item in predict:
                if item not in real:
                    FP += 1
            return TP, FP, FN
    
        a_TP, a_FP, a_FN = 0, 0, 0
        for i, data in enumerate(self.data_test):
            s, target_a = to_device(data)
            a_weights = self.policy(s)
            a = a_weights.ge(0)
            TP, FP, FN = f1(a, target_a)
            a_TP += TP
            a_FP += FP
            a_FN += FN
            
        prec = a_TP / (a_TP + a_FP)
        rec = a_TP / (a_TP + a_FN)
        F1 = 2 * prec * rec / (prec + rec)
        print(a_TP, a_FP, a_FN, F1)

    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_mle.pol.mdl')

        logging.info('<<dialog policy>> epoch {}: saved network to mdl'.format(epoch))

