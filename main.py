import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning))
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import autograd, optim
from comparemodel import UNet,attention_unet ,r2unet,segnet,unetpp,fcn,cenet,LadderNet,DenseUnet
from dataset import *
from metrics import *
from torchvision.transforms import transforms
from plot import loss_plot
from plot import metrics_plot



os.environ["CUDA_VISIBLE_DEVICE"] = "0"
def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--deepsupervision', default=0)
    parse.add_argument("--action", type=str, help="train/test/train&test", default="train&test")
    parse.add_argument("--epoch", type=int, default=4)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='UNet',
                       help='UNet/resnet34_unet/unet++/Attention_UNet/segnet/r2unet/fcn32s/fcn8s/laddernet/denseUnet/Unet3plus/Mycunet/Resunet')
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument('--dataset', default='isbiCell',  # dsb2018_256
                       help='dataset name:liver/esophagus/dsb2018Cell/corneal/driveEye/isbiCell/kaggleLung/poly/skin/Driveaug')
    parse.add_argument("--log_dir", default='result/log', help="log dir")
    parse.add_argument("--threshold", type=float, default=None)
    args = parse.parse_args()
    return args

def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

def getModel(args):
    if args.arch == 'UNet':
        model = UNet.Unet(3, 1).to(device)
    if args.arch == 'resnet34_unet':
        model = UNet.resnet34_unet(1,pretrained=False).to(device)
    if args.arch == 'unet++':
        args.deepsupervision = True
        model = unetpp.NestedUNet(args,3,1).to(device)
    if args.arch =='Attention_UNet':
        model = attention_unet.AttU_Net(3,1).to(device)
    if args.arch == 'segnet':
        model = segnet.SegNet(3,1).to(device)
    if args.arch == 'r2unet':
        model = r2unet.R2U_Net(3,1).to(device)
    # if args.arch == 'fcn32s':
    #     model = get_fcn32s(1).to(device)
    # if args.arch == 'fcn8s':
    #     assert args.dataset !='esophagus' ,"fcn8s模型不能用于数据集esophagus，因为esophagus数据集为80x80，经过5次的2倍降采样后剩下2.5x2.5，分辨率不能为小数，建议把数据集resize成更高的分辨率再用于fcn"
    #     model = get_fcn8s(1).to(device)
    if args.arch == 'cenet':
        model = cenet.CE_Net_().to(device)
    if args.arch == 'laddernet':
        model = LadderNet().to(device)
    if args.arch == 'denseUnet':
        model = DenseUnet.Dense_Unet().to(device)
    if args.arch == 'Unet3plus':
        from model3 import UNet_3Plus
        model =UNet_3Plus.UNet_3Plus().to(device)
    if args.arch == 'DFFnet':
        import model3.DFFnet
        model =model3.DFFnet.Mynet(channel=3).to(device)
    if args.arch == 'DFFnetwithatt':
        import model3.DFFnetwithCSATT
        model = model3.DFFnetwithCSATT.Mynet(channel=3).to(device)
    if args.arch == 'RAS':
        import model3.PraNet_ResNet
        model =model3.PraNet_ResNet.CRANet().to(device)
    if args.arch == 'Resunet':
        import models.core.res_unet_plus
        model =models.core.res_unet_plus.ResUnetPlusPlus(channel=3).to(device)
    return model

def getDataset(args):
    train_dataloaders, val_dataloaders ,test_dataloaders= None,None,None
    if args.dataset =='liver':  #H:\代码\new\u_net_liver-master\data\liver\val
        train_dataset = LiverDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size,drop_last=True)
        val_dataset = LiverDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    if args.dataset =="esophagus":
        train_dataset = esophagusDataset(r"train", transform=x_transforms,target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size,drop_last=True)
        val_dataset = esophagusDataset(r"val", transform=x_transforms,target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    if args.dataset =="skin":
        train_dataset = skinDataset(r"train", transform=x_transforms,target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size,drop_last=True)
        val_dataset = skinDataset(r"val", transform=x_transforms,target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = skinDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == "dsb2018Cell":
        train_dataset = dsb2018CellDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size,drop_last=True)
        val_dataset = dsb2018CellDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataloaders = val_dataloaders
    if args.dataset == 'corneal':
        train_dataset = CornealDataset(r'train',transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size,drop_last=True)
        val_dataset = CornealDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = CornealDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'driveEye':
        train_dataset = DriveEyeDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size,drop_last=True)
        val_dataset = DriveEyeDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = DriveEyeDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'isbiCell':
        train_dataset = IsbiCellDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size,drop_last=True)
        val_dataset = IsbiCellDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = IsbiCellDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'kaggleLung':
        train_dataset = LungKaggleDataset(r'train', transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size,drop_last=True)
        val_dataset = LungKaggleDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = LungKaggleDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'poly':
        train_dataset = ColonpolypsDataset(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size,drop_last=True)
        val_dataset = ColonpolypsDataset(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = ColonpolypsDataset(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    if args.dataset == 'Driveaug':
        train_dataset = DriveEyeAug(r"train", transform=x_transforms, target_transform=y_transforms)
        train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size,drop_last=True)
        val_dataset = DriveEyeAug(r"val", transform=x_transforms, target_transform=y_transforms)
        val_dataloaders = DataLoader(val_dataset, batch_size=1)
        test_dataset = DriveEyeAug(r"test", transform=x_transforms, target_transform=y_transforms)
        test_dataloaders = DataLoader(test_dataset, batch_size=1)
    return train_dataloaders, val_dataloaders, test_dataloaders



def train(model, criterion, optimizer, train_dataloader,val_dataloader, args):
    best_iou, aver_iou, aver_dice, aver_hd,acc = 0,0,0,0,0
    num_epochs = args.epoch
    threshold = args.threshold
    loss_list = []
    iou_list = []
    dice_list = []
    hd_list = []
    acc_list =[]
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0
        step = 0
        for x, y,_,mask in train_dataloader:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            if args.deepsupervision:
                outputs = model(inputs)
                loss = 0
                for output in outputs:
                    loss += criterion(output, labels)
                loss /= len(outputs)
            else:
                output = model(inputs)

                loss = criterion(output, labels)
            if threshold!=None:
                if loss > threshold:
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
            else:
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
            logging.info("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item()))
        loss_list.append(epoch_loss)
        best_iou, aver_iou, aver_dice, aver_hd,acc = val(model, best_iou, val_dataloader)
        iou_list.append(aver_iou)
        dice_list.append(aver_dice)
        hd_list.append(aver_hd)
        acc_list.append(acc)
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    loss_plot(args, loss_list)
    metrics_plot(args, 'iou&dice', iou_list, dice_list)
    metrics_plot(args, 'hd', hd_list)
    return model

def val(model,best_iou,val_dataloaders):
    model.cuda()
    model= model.eval()
    with torch.no_grad():
        i=0   #验证集中第i张图
        miou_total = 0
        hd_total = 0
        dice_total = 0
        acc_total = 0
        num = len(val_dataloaders)#验证集图片的总数
        #print(num)
        for x, _,pic,mask in val_dataloaders:
            x = x.to(device)
            y = model(x)
            if args.deepsupervision:
                img_y = torch.squeeze(y[-1]).cpu().numpy()
            else:
                img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

            hd_total += get_hd(mask[0], img_y)
            miou_total += get_iou(mask[0],img_y)  #获取当前预测图的miou，并加到总miou中,mask在前，predict在后
            dice_total += get_dice(mask[0],img_y)
            acc_total += get_acc(mask[0],img_y)
            if i < num:i+=1   #处理验证集下一张图
        aver_iou = miou_total / num
        aver_hd = hd_total / num
        aver_dice = dice_total/num
        acc = acc_total /num
        print('Miou=%f,aver_hd=%f,aver_dice=%f,acc=%f' % (aver_iou,aver_hd,aver_dice,acc))
        logging.info('Miou=%f,aver_hd=%f,aver_dice=%f,acc=%f' % (aver_iou,aver_hd,aver_dice,acc))
        if aver_iou > best_iou:
            print('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('aver_iou:{} > best_iou:{}'.format(aver_iou,best_iou))
            logging.info('===========>save best model!')
            best_iou = aver_iou
            print('===========>save best model!')
            # state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': args.epoch, 'lr_schedule': scheduler}
            # torch.save(state, r'./saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.dataset) + '_' + str(args.epoch) + '.pth')
            torch.save(model.state_dict(), r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth')
        return best_iou,aver_iou,aver_dice,aver_hd,acc

def test(val_dataloaders,save_predict=False):
    logging.info('final test........')
    if save_predict ==True:
        dir = os.path.join(r'./saved_predict',str(args.arch),str(args.batch_size),str(args.epoch),str(args.dataset))
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('dir already exist!')
    # torch.load(r'./saved_model/'+str(args.arch)+'_'+str(args.batch_size)+'_'+str(args.dataset)+'_'+str(args.epoch)+'.pth', map_location=device)
    model.load_state_dict(torch.load(r'./saved_model/' + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.dataset) + '_' + str(args.epoch) + '.pth', map_location=device))
    model.eval()

    #plt.ion() #开启动态模式
    with torch.no_grad():
        i=0   #验证集中第i张图
        miou_total = 0
        hd_total = 0
        dice_total = 0
        acc_total = 0
        num = len(val_dataloaders)  #验证集图片的总数
        for pic,_,pic_path,mask_path in val_dataloaders:
            pic = pic.to(device)
            predict = model(pic)
            if args.deepsupervision:
                predict = torch.squeeze(predict[-1]).cpu().numpy()
            else:
                predict = torch.squeeze(predict).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize
            #img_y = torch.squeeze(y).cpu().numpy()  #输入损失函数之前要把预测图变成numpy格式，且为了跟训练图对应，要额外加多一维表示batchsize

            iou = get_iou(mask_path[0],predict)
            miou_total += iou  #获取当前预测图的miou，并加到总miou中
            hd_total += get_hd(mask_path[0], predict)
            dice = get_dice(mask_path[0],predict)
            dice_total += dice
            acc = get_acc(mask_path[0], predict)
            acc_total += acc
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 3, 1)
            #去轴合并空白
            plt.axis('off')
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            # plt.margins(0, 0)
            ax1.set_title('input')
            plt.imshow(Image.open(pic_path[0]).convert('RGB'))
            #print(pic_path[0])
            ax2 = fig.add_subplot(1, 3, 2)
            # 去轴合并空白
            plt.axis('off')
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            # plt.margins(0, 0)
            ax2.set_title('predict')
            plt.imshow(predict,cmap='Greys_r')
            ax3 = fig.add_subplot(1, 3, 3)
            # 去轴合并空白
            plt.axis('off')
            # plt.gca().xaxis.set_major_locator(plt.NullLocator())
            # plt.gca().yaxis.set_major_locator(plt.NullLocator())
            # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            # plt.margins(0, 0)
            ax3.set_title('mask')
            plt.imshow(Image.open(mask_path[0]), cmap='Greys_r')
            #print(mask_path[0])
            if save_predict == True:
                if args.dataset == 'driveEye':
                    saved_predict = dir + '/' + mask_path[0].split('\\')[-1]
                    saved_predict = '.'+saved_predict.split('.')[1] + '.png'
                    # plt.savefig(saved_predict, bbox_inches='tight', dpi=fig.dpi)
                    plt.savefig(saved_predict,bbox_inches='tight', dpi=fig.dpi)
                else:
                    plt.savefig(dir +'/'+ mask_path[0].split('\\')[-1], bbox_inches='tight', dpi=fig.dpi)
            #plt.pause(0.01)
            print('iou={},dice={}'.format(iou,dice))
            plt.close("all")
            if i < num:i+=1   #处理验证集下一张图
        #plt.show()
        print('Miou=%f,aver_hd=%f,dice=%f,acc=%f' % (miou_total/num,hd_total/num,dice_total/num,acc_total/num))
        logging.info('Miou=%f,aver_hd=%f,dice=%f,acc=%f' % (miou_total/num,hd_total/num,dice_total/num,acc_total/num))

if __name__ =="__main__":

    x_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # mask只需要转换为tensor
    y_transforms = transforms.ToTensor()

    # x_transforms = transforms.Compose([
    #     transforms.ToPILImage(model=3),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation((-10, 10)),
    #     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),#mask 对应 Grayscale()
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    # ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = getArgs()
    logging = getLog(args)
    print('**************************')
    print('model3:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodel3:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
          (args.arch, args.epoch, args.batch_size, args.dataset))
    print('**************************')
    model = getModel(args)
    train_dataloaders, val_dataloaders, test_dataloaders = getDataset(args)
    criterion = torch.nn.BCELoss()

    initial_lr = 0.0001
    # # opt = torch.optim.Adam(model_test.parameters(), lr=initial_lr)  # try SGD
    # # # opt = optim.SGD(model_test.parameters(), lr = initial_lr, momentum=0.99)
    optimizer = optim.Adam(model.parameters(),lr=initial_lr,weight_decay=1e-5)
    MAX_STEP = int(1e10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MAX_STEP, eta_min=1e-6)
    # optimizer = optim.Adam(model.parameters())
    if 'train' in args.action:
        train(model, criterion, optimizer, train_dataloaders,val_dataloaders, args)
    if 'test' in args.action:
        test(test_dataloaders, save_predict=True)