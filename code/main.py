import os
import cv2
import time
import models
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optmizer
from torchnet import meter
import torch.nn as nn
from dataset import Dataset
from config import DefaultConfig
from utils import Visualizer, SSIM
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
opt = DefaultConfig()
dataset = Dataset(opt)

def test (**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)
    typelist = ['radar','wind','precip']
    Dmodellist =["../user_data/Dradar.pth","../user_data/Dwind.pth","../user_data/Dprecip.pth"]
    Gmodellist =["../user_data/Gradar.pth","../user_data/Gwind.pth","../user_data/Gprecip.pth"]
    time_now = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    for opt.datatype,opt.load_model_path,opt.load_G_nodel_path in zip(typelist,Dmodellist,Gmodellist):
        #step1: 模型
        model = getattr (models,opt.model)
        if opt.model == 'DNet':
            G_model = getattr(models,'UNet')
        if opt.use_gpu :
            model = model(16,6).cuda()
            G_model = G_model().cuda()
        else: 
            model = model(16,6)
            G_model = G_model()
        if opt.load_model_path: 
            model.load(path = opt.load_model_path)
        if opt.load_G_model_path: 
            G_model.load(path = opt.load_G_model_path)

        else:
            print('Please input the model file!')
            exit()
        #step2: 数据
        new = {'phase':'test'}
        opt.parse(new)
        test_data = Dataset(opt)
        test_dataloader = torch.utils.data.DataLoader(test_data,opt.batch_size,
                shuffle = False,
                num_workers = opt.num_workers)
        criterion2 = nn.MSELoss()
        loss_meter = meter.AverageValueMeter()
        for ii,data in enumerate(tqdm(test_dataloader)):
            #训练模型参数
            data,white,fold = data
            input = Variable(data.type(torch.float32),volatile = True)
            if opt.use_gpu:input = input.cuda()
            output = G_model(input)

            out = output.detach().cpu().numpy()[0]
            fold = os.path.join(opt.savepath.format('submit_' + time_now,opt.datatype.title()),fold[0])
            if os.path.exists(fold) == False:os.makedirs(fold)
            f,h,w = out.shape
            if white>0:out = out[:,white:h-white,:]
            else:out = out[:,:,white:h-white]
            out = (255*out)
            for i in range(out.shape[0]):
                cv2.imwrite(os.path.join(fold,opt.datatype.lower() + '_' + str(i+1).zfill(3) + '.png'),out[i] ,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])



def train (**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)
    #step1: 模型
    model = getattr (models,opt.model)
    if opt.model == 'DNet':
        G_model = getattr(models,'UNet')
    if opt.use_gpu :
        model = model(16,6).cuda()
        G_model = G_model().cuda()
    else:
        model = model(16,6)
        G_model = G_model()
    if opt.load_model_path: 
        model.load(path = opt.load_model_path)
    if opt.load_G_model_path: 
        G_model.load(path = opt.load_G_model_path)
    #step2: 数据

    train_data = Dataset(opt)
    new = {'phase':'val'}
    opt.parse(new)
    val_data = Dataset(opt)
    new = {'phase':'train'}
    opt.parse(new)
    train_dataloader = torch.utils.data.DataLoader(train_data,opt.batch_size,
            shuffle = True,
            drop_last = True,
            num_workers = opt.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_data,opt.batch_size,
            shuffle = False,
            num_workers = opt.num_workers)
    
    #step3: 目标函数和优化器
    #criterion2 = PointLoss()
    criterion1 = nn.L1Loss()
    criterion2 = nn.MSELoss()
    #criterion2 = SSIM()
    #criterion = nn.MSELoss()
    #criterion = SortLoss()
    #criterion2 = HauLoss()
    


    lr = opt.lr
    D_optimizer = optmizer.Adam(model.parameters(),lr = 0.0002,betas=[0.5,0.999])
    G_optimizer = optmizer.Adam(G_model.parameters(),lr = 0.0002,betas=[0.5,0.999])
    g_lr_decay = optmizer.lr_scheduler.StepLR(G_optimizer,step_size=opt.max_epoch,gamma=0.1)
    d_lr_decay = optmizer.lr_scheduler.StepLR(D_optimizer,step_size=opt.max_epoch,gamma=0.1)
    #step4: 统计指标：平滑处理后的损失，还有混淆矩阵

    g_loss_meter = meter.AverageValueMeter()
    d_loss_meter = meter.AverageValueMeter()
    #confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10
    val_loss=[]
    #训练
    data = next(iter(train_dataloader)).type(torch.float32)
    input = data[:,:20]
    target = data[:,20:-1]
    #fix = model(input[:,0].cuda(),input[:,0].cuda())
    fix = model(input.cuda(),input.cuda())
    ones = torch.ones_like(fix)
    zeros = torch.zeros_like(fix)
    for epoch in range(opt.max_epoch):
        g_loss_meter.reset()
        d_loss_meter.reset()
        #confusion_matrix.reset()
        g_lr_decay.step(epoch)
        d_lr_decay.step(epoch)
        for ii,data in enumerate(tqdm(train_dataloader)):
            #训练模型参数
            data = Variable(data.type(torch.float32),requires_grad=True)
            if opt.use_gpu:
                data = data.cuda()
            input = data[:,:20]
            target = data[:,-20:]
            ###################################  GNet #################################################
            fake = G_model(input)
            gan_loss = criterion2(model(fake,input),ones)
            recon_loss = criterion1(fake,target)
            g_loss = gan_loss+100*recon_loss
            G_optimizer.zero_grad()
            g_loss.backward(retain_graph=True)
            G_optimizer.step()
            ###################################  DNet #################################################
            loss_real = criterion2(model(target,input),ones)
            loss_fake = criterion2(model(fake.detach(),input),zeros)
            d_loss = (loss_real+loss_fake)/2.0
            D_optimizer.zero_grad()
            d_loss.backward() 
            D_optimizer.step()

            #更新统计指标及可视化
            g_loss_meter.add(g_loss.data.cpu())
            d_loss_meter.add(d_loss.data.cpu())
            #confusion_matrix.add(score.data,target.data)
            #如果需要的话，进入debug模式
                
            if os.path.exists(opt.debug_file):
                import ipdb
                ipdb.set_trace()
        if opt.model_save :
            model.save()
            G_model.save()
        
        if opt.val == True:
            #计算验证集上的指标及可视化
            #val_cm,val_accuracy = val(model,val_dataloader)
            d_loss_meter,g_loss_meter,fake,target = val(G_model, model,val_dataloader,ones,zeros)
            vis.plot('g_loss',g_loss_meter.value()[0])
            vis.plot('d_loss',d_loss_meter.value()[0])
            vis.imag('output_image',(255*fake[0].detach().unsqueeze(1)))
            vis.imag('target_image',(255*target[0].detach()))
            vis.log("epoch:{epoch},lr:{lr},g_loss:{g_loss},d_loss:{d_loss}".format(epoch = epoch,d_loss = d_loss_meter.value()[0],g_loss = g_loss_meter.value()[0], lr = lr))


def val (G_model,model,dataloader,ones,zeros):

    '''
    计算模型在验证集上的性能指标
    '''

    #把模式改为验证模式
    G_model.eval()
    model.eval()
    criterion1 = nn.L1Loss()
    criterion2 = nn.MSELoss()
    #criterion2 = SSIM()
    vgloss_meter = meter.AverageValueMeter()
    vdloss_meter = meter.AverageValueMeter()
    for ii, data in enumerate(tqdm(dataloader)):
        data = Variable(data.type(torch.float32),volatile = True)
        if opt.use_gpu:
            data = data.cuda()
        input = data[:,:20]
        target = data[:,-20:]
        ###################################  GNet #################################################
        fake = G_model(input)
        gan_loss = criterion2(model(fake,input),ones)
        recon_loss = criterion1(fake,target)
        g_loss = gan_loss+100*recon_loss
        ###################################  DNet #################################################
        loss_real = criterion2(model(target,input),ones)
        loss_fake = criterion2(model(fake.detach(),input),zeros)
        d_loss = (loss_real+loss_fake)/2.0
        vgloss_meter.add(g_loss.data.cpu())
        vdloss_meter.add(d_loss.data.cpu())
    #把模型恢复为训练模式
    model.train()

    return vdloss_meter,vgloss_meter,fake,target




if __name__ =='__main__':
    import fire
    fire.Fire()
