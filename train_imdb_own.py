from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random

from pytorch_transformers import AdamW

from mixtext import MixText
import dataloader_imdb as dataloader



parser = argparse.ArgumentParser(description='IMDB Training')
parser.add_argument('--gpu', default='0,1', type=str)
parser.add_argument('--n_labels', default=2, type=int)
parser.add_argument('--mix_option', default=True, type=bool, help='mix option, whether to mix or not')
parser.add_argument('--data_path', default='/data/zhuoer/IMDB', type=str, help='path to dataset')
parser.add_argument('--dataset', default='imdb', type=str)
parser.add_argument('--r', default=0.2, type=float, help='noise ratio')
parser.add_argument('--noise_mode',  default='asym')
parser.add_argument('--num_epochs', default=15, type=int)
parser.add_argument('--warm_up', default=2, type=int)
parser.add_argument('--batch_size', default=8, type=int)

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--alpha', default=0.75, type=float,help='alpha for beta distribution')
parser.add_argument('--mix-layers-set', nargs='+',
                    default=[7, 9, 12], type=int, help='define mix layer set')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')

parser.add_argument('--un_aug', default=True, type=bool, help='augment unlabeled training data')
args = parser.parse_args()


print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("gpu num: ", n_gpu)


test_log=open('/output/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_text_acc.txt','w')
#loss_log=open('../checkpoints/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_train_loss.txt','w')

def create_model():
    model = MixText(args.n_labels, args.mix_option).cuda()
    model = nn.DataParallel(model)
    return model

CEloss = nn.CrossEntropyLoss()

def warmup(epoch,net,optimizer,dataloader): #一个网络CE
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1   #//除法取整 总iter
                                                                        ##能整除不就多了一个
    correct = 0
    loss_total = 0
    total_sample = 0
    for batch_idx, (inputs, labels, length) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        outputs = net(inputs)
        loss = CEloss(outputs, labels)   #cross entropy #SOFTMAX

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)

        correct += (np.array(predicted.cpu()) ==
                    np.array(labels.cpu())).sum()
        loss_total += loss.item() * inputs.shape[0]
        total_sample += inputs.shape[0]

        sys.stdout.write('\r') #一个 batch 打印一下
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch+1, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

    #print('\n%s:%.1f | Epoch [%3d/%3d]\t ACC:%.2f LOSS:%.2f'% (args.dataset, args.r, epoch+1, args.num_epochs,\
                                                             #correct/total_sample,loss_total/total_sample))



def test(epoch, net1, net2, test_loader):
    net1.eval()
    net2.eval()

    correct1 = 0
    correct2 = 0
    correct = 0

    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets,length) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)
            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)

            outputs = outputs1 + outputs2
            _, predicted = torch.max(outputs, 1)

            total += targets.size(0)
            correct1 += predicted1.eq(targets).cpu().sum().item()
            correct2 += predicted2.eq(targets).cpu().sum().item()
            correct += predicted.eq(targets).cpu().sum().item()

    acc1 = 100. * correct1 / total
    acc2 = 100. * correct2 / total
    #acc=(acc1+acc2)/2
    acc_sum = 100. * correct / total

    print("\n| Test:Epoch #%d\t Accuracy1: %.2f%%" % (epoch + 1, acc1))
    print("| Test:Epoch #%d\t Accuracy2: %.2f%%" % (epoch + 1, acc2))
    print("| Test:Epoch #%d\t Accuracy: %.2f%%" % (epoch + 1, acc_sum))
    #print("| Test:Epoch #%d\t Accuracy: %.2f%%" % (epoch + 1, acc))
    test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch + 1, acc_sum))
    #test_log.write('Epoch:%d   Accuracy:%.2f\n' % (epoch + 1, acc))
    test_log.flush()
    return acc_sum


def eval_train(net,eval_loader):
    net.eval()

    pred_sum = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, length) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)

            pred = predicted.eq(targets).cpu().numpy().astype(int)
            pred_sum.extend(pred.tolist())

    return pred_sum

def linear_rampup(current, warm_up, rampup_length=16): #current epoch
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

def train(epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader):
    net.train()
    net2.eval()  # fix one network and train the other

    unlabeled_train_iter = iter(unlabeled_trainloader)
    num_iter = (len(labeled_trainloader.dataset) // args.batch_size) + 1

    for batch_idx, (inputs_x,  labels_x, _) in enumerate(labeled_trainloader):
        if args.un_aug:
            try:
                inputs_u, inputs_u2 = next(unlabeled_train_iter)
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, inputs_u2 = next(unlabeled_train_iter)
        else:
            try:
                inputs_u  = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u  = (unlabeled_train_iter).next()
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.n_labels).scatter_(1, labels_x.view(-1, 1), 1)

        inputs_x, labels_x = inputs_x.cuda(), labels_x.cuda()
        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u1 = net(inputs_u)
            outputs_u2 = net2(inputs_u)
            outputs_u3 = net(inputs_u2)
            outputs_u4 = net2(inputs_u2)

            pu = (torch.softmax(outputs_u1, dim=1)  + torch.softmax(outputs_u2,dim=1) + torch.softmax(outputs_u3, dim=1) + torch.softmax(outputs_u4, dim=1) )/ 4
            ptu = pu ** (1 / args.T)  # temparature sharpening
            targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
            targets_u = targets_u.detach()

            targets_x = labels_x.detach()

        # TMIX
        mix_layer = np.random.choice(args.mix_layers_set, 1)[0]  # 从 mix_layers_set 里随机产生一个层数
        mix_layer = mix_layer - 1

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1 - l)

        all_inputs = torch.cat(
            [inputs_x, inputs_u, inputs_u2], dim=0)

        all_targets = torch.cat(
            [targets_x, targets_u, targets_u], dim=0)

        idx = torch.randperm(all_inputs.size(0))  # 0-all_inputs.size(0)-1打乱全排
        ###多加点clean+clean
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        logits = net(input_a, input_b, l, mix_layer)
        mixed_target = l * target_a + (1 - l) * target_b

          # batch_size = inputs_x.size(0)

        Lx, Lu, lamb = criterion(logits[:batch_size], mixed_target[:batch_size], logits[batch_size:], mixed_target[batch_size:],epoch + batch_idx / num_iter, args. warm_up)

        prior = torch.ones(args.n_labels) / args.n_labels
        prior = prior.cuda()
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))
        loss = Lx + lamb * Lu + penalty ###

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # 更新 Net

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                         % (args.dataset, args.r, args.noise_mode, epoch+  1, args.num_epochs, batch_idx + 1, num_iter,
                            Lx.item(), Lu.item()))
        sys.stdout.flush()

        # loss_log.write('%s:%.1f | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f\n'
        #                  % (args.dataset, args.r,  epoch+ 1, args.num_epochs, batch_idx + 1, num_iter,
        #                     Lx.item(), Lu.item()))


loader = dataloader.imdb_dataloader(data_path=args.data_path,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=16,\
    noise_file='%s/noise_ratio= %.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net1 = create_model()
net2 = create_model()
cudnn.benchmark = True

criterion = SemiLoss()
optimizer1 = AdamW(
        [
            {"params": net1.module.bert.parameters(), "lr": args.lrmain},
            {"params": net1.module.linear.parameters(), "lr": args.lrlast},
        ])
optimizer2 = AdamW(
        [
            {"params": net2.module.bert.parameters(), "lr": args.lrmain},
            {"params": net2.module.linear.parameters(), "lr": args.lrlast},
        ])

torch.cuda.empty_cache()
best_acc = 0
best_epoch = 0
for epoch in range (args.num_epochs):

    eval_loader = loader.run('eval_train')
    test_loader = loader.run('test')

    if epoch < args.warm_up:
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, warmup_trainloader)
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, warmup_trainloader)

    else:
        pred_sum1 = eval_train(net1,eval_loader)
        pred_sum2 = eval_train(net2, eval_loader)

        print('Train Net1')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred_sum1,args.un_aug)
        train(epoch, net1, net2, optimizer1, labeled_trainloader, unlabeled_trainloader)  # train net1

        print('\nTrain Net2')
        labeled_trainloader, unlabeled_trainloader = loader.run('train', pred_sum2,args.un_aug)
        train(epoch, net2, net1, optimizer2, labeled_trainloader, unlabeled_trainloader)  # train net2
        ###warm up的时候要aug吗？试一下

    current_acc = test(epoch, net1, net2,test_loader)
    if current_acc > best_acc:
         best_acc = current_acc
         best_epoch = epoch + 1

print(best_acc,best_epoch)




