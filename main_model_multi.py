import argparse, os
import time
import torch
import random
import shutil
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util import AverageMeter
from util import cal_psnr,cal_ssim
from util import save_image,save_image_single
#from loss import SSIM, PAMSE

#from model_dncnn import Model
#from  model_mdr import Model 
#from model_multi_single import Model
#from densenet import DenseNet121
#from hazy_model import Model
from model_multi import Model
#from model_rdn import Model
from torch.optim.lr_scheduler import MultiStepLR
from custom_transform import SingleRandomCropTransform
from custom_transform import SingleTransform
#from tensorboard_logger import log_value, configure
from dataset import CustomDataset
from collections import OrderedDict
import data_generator as dg
from data_generator import DenoisingDataset
import torchvision.models as models
from tensorboardX import SummaryWriter
from est import Model as est_noise
#import visdom

#vis = visdom.Visdom(env='test')
# Training and Testing settings
parser = argparse.ArgumentParser(description="PyTorch denoise Experiment")
parser.add_argument('--ck', dest='ckpt_dir', default='./checkpoint', help="Models are saved here")
parser.add_argument('--result_dir', dest='result_dir', default='./result', help='Test image result dir')
parser.add_argument('--train_dir', dest='train_dir',default='/home/xcy/dataset/dataset_denoise/BSD300/', help="Train data dir")
#parser.add_argument('--train_dir', dest='train_dir', default='./data/train', help="Train data dir")
parser.add_argument('--test_dir', dest='test_dir', default='/home/xcy/dataset/dataset_denoise/test_set/', help='Test data dir')
parser.add_argument('--patch_size', type=int, default=40, help='Training patch size')
parser.add_argument('--noise_level', type=int, default=25, help='Noise level')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size") #origin 64 changed by xcy @20191106
parser.add_argument("--nEpochs", type=int, default=100, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=20, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument("--snr", default=0.95, type=float, help="dropout param setting")
parser.add_argument("--n_type", default=1, type=int, help="noise type")
parser.add_argument("--input_number", default=3, type=int, help="noise type")

parser.add_argument("--lamda", default=1, type=float, help="dropout param setting")

#tensorboard setting
writer = SummaryWriter('tensorboard/%d'%(time.time()))
best_accuracy = 0
def main():
    global opt, best_accuracy
    opt = parser.parse_args()
    print(opt)

    if not os.path.exists(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir)
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
    if not os.path.exists(opt.train_dir):
        print("{} not exist".format(opt.train_dir))
        return
    if not os.path.exists(opt.test_dir):
        print("{} not exist".format(opt.test_dir))
        return
    #configure(os.path.join(opt.ckpt_dir, 'log'), flush_secs=5)

    #cuda = False
    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    #opt.seed = random.randint(1, 10000)
    opt.seed = 0
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    '''
    train_transform = SingleRandomCropTransform(opt.patch_size,
                                                opt.noise_level)
    train_set = CustomDataset(opt.train_dir, train_transform)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    '''

    test_transform = SingleTransform(opt.noise_level,noise_type = opt.n_type)
    test_set = CustomDataset(opt.test_dir, test_transform)
    test_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size = 1, shuffle=False)
    print("===> Building model")
    #model = Model(image_channels = opt.input_number)
    model = Model()
    #model = DenseNet121();
    print(model)
    criterion = nn.MSELoss(reduction='sum')
    #criterion = nn.L1Loss(reduction='sum')
    #criterion = PAMSE()
    #criterion = nn.L1Loss(size_average=False)
    print("===> Setting GPU")
    if cuda:
        #model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        print("Not Using GPU")
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            checkpoint = torch.load(opt.resume)
            
            model_dict = model.state_dict()
            checkpoint_load = {k: v for k, v in (checkpoint['model']).items() if k in model_dict}
            model_dict.update(checkpoint_load)
            model.load_state_dict(model_dict)

            '''
            pretrained_net_dict = checkpoint["model"]

            new_state_dict = OrderedDict()
            for k, v in pretrained_net_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
  # load params
            '''
            print("=> loading checkpoint '{}'".format(opt.resume))
            opt.start_epoch = checkpoint["epoch"] + 1
            print("=> start_epoch set to '{}'".format(opt.start_epoch))
            #opt.start_epoch = 1
            best_accuracy = checkpoint['best_accuracy']
            #model.load_state_dict(new_state_dict)
            #model.load_state_dict(checkpoint['model'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            checkpoint = torch.load(opt.pretrained)
            
            model_dict = model.state_dict()
            checkpoint_load = {k: v for k, v in (checkpoint['model']).items() if k in model_dict}
            model_dict.update(checkpoint_load)
            model.load_state_dict(model_dict)

            '''
            pretrained_net_dict = checkpoint["model"]

            new_state_dict = OrderedDict()
            for k, v in pretrained_net_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
  # load params
            '''
            print("=> loading checkpoint '{}'".format(opt.resume))
            #opt.start_epoch = checkpoint["epoch"] + 1
            #opt.start_epoch = 1
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    if opt.evaluate:
        if opt.resume:
            validate(test_data_loader, model, criterion, opt.start_epoch)
            return
        else:
            print("!!!!!!!!!!!!!!!!!! please choose a resume model !!!!!!!!!!")
            return

    xs = dg.datagenerator(data_dir=opt.train_dir)
    xs = xs.astype('float32')/255.0
    xs = torch.from_numpy(xs.transpose((0,3,1,2)))
    train_set = DenoisingDataset(xs,opt.noise_level)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, drop_last=True,shuffle=True)
    print("===> Setting Optimizer")
    #optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    #optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    #for param in model.module.part1.parameters():
    #    param.require_grad = False
    
    #ignored_params1 = list(map(id, model.module.option1.parameters()))
    #ignored_params2 = list(map(id, model.module.option2.parameters()))
    #ignored_params3 = list(map(id, model.module.option3.parameters()))
    #ignored_params  = ignored_params1 + ignored_params2+ignored_params3
    #base_params = filter(lambda p: id(p) not in ignored_params,
    #                     model.module.parameters())
    #adjust_param = filter(lambda p: id(p)  in ignored_params,
    #                     model.module.parameters())
    #optimizer = torch.optim.Adam([
    #                 {'params': base_params},
    #                 {'params': adjust_param,'lr':0.00001} ], lr=0, weight_decay = opt.weight_decay)
   
    print("===> Training")
    step_1 = int(opt.nEpochs*3/10)
    step_2 = int(opt.nEpochs*6/10)
    step_3 = int(opt.nEpochs*9/10)
    print(opt.snr)
    scheduler = MultiStepLR(optimizer,milestones=[step_1,step_2,step_3],gamma=0.25)
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        scheduler.step(epoch)
        print("===>Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
        train(training_data_loader, optimizer, model, criterion, epoch)

        is_best = 0
        if epoch%2 == 1:
            psnr = validate(test_data_loader, model, criterion, epoch)
            is_best = psnr > best_accuracy
            best_accuracy = max(psnr, best_accuracy)
            save_checkpoint({'epoch': epoch,
                         'best_accuracy':best_accuracy,
                         'model': model.state_dict()}, is_best, epoch)
    #save_checkpoint({'epoch': opt.nEpochs,
    #                 'best_accuracy':30,
    #                 'model': model.state_dict()}, 0, opt.nEpochs)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    #lr = opt.lr * (0.1 ** (epoch // 40))
    lr = opt.lr
    if epoch == 40:
        lr = opt.lr*0.1
    elif epoch == 60:
        lr = opt.lr*0.01
    elif epoch == 80:
        lr = opt.lr*0.001
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train(training_data_loader, optimizer, model, criterion, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    avg_psnr = AverageMeter()
    global Writer
    model.train()
    end = time.time()
    for i, batch in enumerate(training_data_loader, 1):
        data_time.update(time.time() - end)
        noise_image, groundtruth= batch[0],batch[1]
        if opt.cuda:
            noise_image = noise_image.cuda()
            groundtruth = groundtruth.cuda()
        noise_level = batch[3].view(-1,1,1,1).type_as(batch[0])
        noise_level = torch.ones(noise_image.size()).mul_(noise_level)
        noise_level = noise_level.cuda()


        noise_type = batch[2].view(-1,1,1,1).type_as(batch[0])
        noise_type = torch.ones(noise_image.size()).mul_(noise_type)
        noise_type = noise_type.cuda()

        noise_image.requires_grad_()
        groundtruth.requires_grad_(False)
        if opt.input_number == 3:
            mul_input = torch.cat((noise_level,noise_type,noise_image),1)
        else:
        	mul_input = torch.cat((noise_level,noise_image),1)
        clean_image,_ = model(mul_input)
        
        loss = criterion(clean_image, groundtruth)/(noise_image.size()[0]*2)
        
        #loss = criterion(clean_image, groundtruth)/(noise_image.size()[0]*2)
        losses.update(loss.item(), clean_image.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ground_truth = torch.clamp(255 * groundtruth, 0, 255).byte()
        output_clean_image = torch.clamp(255 * clean_image, 0, 255).byte()
        psnr = cal_psnr(ground_truth.data.cpu().numpy(), output_clean_image.data.cpu().numpy())
        avg_psnr.update(psnr, noise_image.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Psnr {psnr.val:.3f} ({psnr.avg:.3f})'.format(
                   epoch, i, len(training_data_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, psnr=avg_psnr))
    writer.add_scalar("train_loss",losses.avg,epoch)
    writer.add_scalar('train_avg_psnr', avg_psnr.avg, epoch)
    #log_value('train_loss', losses.avg, epoch)
    #log_value('train_avg_psnr', avg_psnr.avg, epoch)

def validate(test_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    avg_psnr = AverageMeter()
    avg_ssim = AverageMeter()

    model.eval()

    global writer
    with torch.no_grad():
        for i,(image, target,noise_type,noise_level) in enumerate(test_loader):
            image_var = image
            target_var = target

            if opt.cuda:
                image_var = image_var.cuda()
                target_var = target_var.cuda()


            noise_level = noise_level.view(-1,1,1,1).type_as(image)
            noise_level = torch.ones(image_var.size()).mul_(opt.lamda*noise_level)
            noise_level = noise_level.cuda()


            noise_type = noise_type.view(-1,1,1,1).type_as(image)
            noise_type = torch.ones(image_var.size()).mul_(noise_type)
            noise_type = noise_type.cuda()

            end = time.time()
            if opt.input_number == 3:
                mul_input = torch.cat((noise_level,noise_type,image_var),1)
            else:
                mul_input = torch.cat((noise_level,image_var),1)
            clean_image,_ = model(mul_input)
            #_,_,w,h = p.size()
            #if i == 1:
            #    vis.images(p.view(-1,1,w,h).data.cpu().numpy(), nrow=8, win="window")
            #clean_image = model(image_var)
            batch_time.update(time.time() - end)
         
            loss = criterion(clean_image, target_var)
            losses.update(loss.item(), image_var.size(0))
            ground_truth = torch.clamp(255 * target_var, 0, 255).byte()
            output_image = torch.clamp(255 * clean_image, 0, 255).byte()
            noise_image = torch.clamp(255 * image_var, 0, 255).byte()
           
            #save_image(ground_truth.data.cpu().numpy(), noise_image.data.cpu().numpy(), output_image.data.cpu().numpy(),os.path.join(opt.result_dir, 'test%d.png'%i))
            save_image_single(ground_truth.data.cpu().numpy(), noise_image.data.cpu().numpy(), output_image.data.cpu().numpy(),os.path.join(opt.result_dir, 'test%d.png'%i))
            psnr = cal_psnr(ground_truth.data.cpu().numpy(), output_image.data.cpu().numpy())
            ssim = cal_ssim(ground_truth.data.cpu().numpy(), output_image.data.cpu().numpy())
        #output_image = torch.clamp(255 * clean_image_hr, 0, 255).byte()
        #psnr1 = cal_psnr(ground_truth.data.cpu().numpy(), output_image.data.cpu().numpy())
        #print("psnr hr %.3f!!!!-----------psnr lr %.3f!!!!!!!!!-"%(psnr,psnr1))
        #save_image(ground_truth.data.cpu().numpy(), noise_image.data.cpu().numpy(),
        #        output_image.data.cpu().numpy(),os.path.join(opt.result_dir, 'test_lr%d.png'%i))

            avg_psnr.update(psnr, image_var.size(0))
            avg_ssim.update(ssim, image_var.size(0))

            if i % 1 == 0:
                print('Test Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                  'Ssim {ssim.val:.4f} ({ssim.avg:.4f})\t'
                  'Psnr {psnr.val:.3f} ({psnr.avg:.3f})'.format(
                   epoch, i, len(test_loader), batch_time=batch_time,
                   ssim=avg_ssim, psnr=avg_psnr))

    #log_value('test_loss', losses.avg, epoch)
    #log_value('test_avg_psnr', avg_psnr.avg, epoch)

    print("--- Epoch %d  --------- Average PSNR %.3f ---" %(epoch, avg_psnr.avg))

    value_out_path = os.path.join(opt.ckpt_dir, "num.txt")
    F = open(value_out_path,'a')
    F.write("Epoch %d: PSNR %.3f ssim %.4f\n"%(epoch,avg_psnr.avg,avg_ssim.avg))
    F.close()

    return avg_psnr.avg

def save_checkpoint(state, is_best, epoch):
    model_out_path = os.path.join(opt.ckpt_dir, "model_epoch_{}.pth".format(epoch))
    torch.save(state, model_out_path)
    #print("Checkpoint saved to {}".format(model_out_path))
    if is_best:
        best_model_name = os.path.join(opt.ckpt_dir, "model_best.pth")
        shutil.copyfile(model_out_path, best_model_name)
        print('Best model {} saved to {}'.format(model_out_path, best_model_name))

if __name__ == "__main__":
    main()
