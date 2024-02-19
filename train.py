from dataset.dataset import *
from model.model import Model
import argparse
import random
import albumentations
from utils import Logger
from losses import *

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------- Arguments ---------------
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--batch-size-test', type=int, default=15)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--checkpoint_dir', default='./train_result/model_k', type=str)
parser.add_argument('--scene-directory-train', default='./data/Training', type=str)
#parser.add_argument('--scene-directory-train', default='./data/synhdr_train', type=str)
args = parser.parse_args()

# --------------- Loading ---------------
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    setup_seed(42)
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.xavier_uniform_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),  # 水平翻转，翻转概率为0.5
        albumentations.VerticalFlip(p=0.5),
        ])

    train_data = MyDataset(traindata_path=args.scene_directory_train, transforms=transforms, noise=None, is_training=True)  # 有getitem变成可迭代的对象
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=False)

    model = Model().to(device)
    L1_loss = nn.L1Loss()
    Get_gradient = Get_gradient()
    model.apply(weights_init_normal)

    optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = WarmupMultiStepLR(optimizer=optimizer_model, milestones=(2000, 4000), gamma=0.1,
                                  warmup_factor=1.0 / 3, warmup_iters=200, warmup_method="linear", last_epoch=-1)
    counter = 0
    counter_test = 0
    Tensor = torch.cuda.FloatTensor
    logger = Logger(os.path.join('./mybase//log', 'LR_log.txt'))

    start_epoch = 0
    checkpoint_file = args.checkpoint_dir + '/model.pkl'
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_model.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("Load checkpoint %s (epoch %d)", checkpoint_file, start_epoch)

    global cur_epoch
    for epoch in range(start_epoch, 6001):
        cur_epoch = epoch
        print('**************** Epoch %d ****************' % (epoch + 1))
        print('learning rate: %f' % (scheduler.get_last_lr()[0]))
        # scheduler.step()

        # logger("第{}个epoch的学习率：{:.7f}".format(epoch, optimizer_model.param_groups[0]['lr']))
        for step, sample in enumerate(train_loader, 0):
            a = time.time()

            batch_x1, batch_x2, batch_x3, batch_x4, batch_xs = \
                sample['input1'], sample['input2'], sample['input3'], sample['label'], sample['Ms']
            batch_x1, batch_x2, batch_x3, batch_x4, batch_xs = \
                batch_x1.to(device), batch_x2.to(device), batch_x3.to(device), batch_x4.to(device), batch_xs.to(device)

            optimizer_model.zero_grad()
            model.train()
            out1, out2 = model(batch_x1, batch_x2, batch_x3, batch_xs)
            lable_gradient_map = Get_gradient(batch_x4)
            out1_gradient_map = Get_gradient(out1)
            out2_gradient_map = Get_gradient(out2)

            PSNR1 = batch_PSNR(torch.clamp(out1, 0., 1.), batch_x4, 1.)
            SSIM1 = batch_SSIM(torch.clamp(out1, 0., 1.), batch_x4, 1.)
            pre_compressed1 = range_compressor_tensor(out1)
            batch_x4_compressed = range_compressor_tensor(batch_x4)
            PSNR_u1 = batch_PSNR(torch.clamp(pre_compressed1, 0., 1.), batch_x4_compressed, 1.)
            SSIM_u1 = batch_SSIM(torch.clamp(pre_compressed1, 0., 1.), batch_x4_compressed, 1.)
            total_loss_train1 = L1_loss(pre_compressed1, batch_x4_compressed)  + 0.5*L1_loss(out1_gradient_map, lable_gradient_map) + 0.2*(1-SSIM_u1) 
                               


            PSNR = batch_PSNR(torch.clamp(out2, 0., 1.), batch_x4, 1.)
            SSIM = batch_SSIM(torch.clamp(out2, 0., 1.), batch_x4, 1.)
            pre_compressed = range_compressor_tensor(out2)
            batch_x4_compressed = range_compressor_tensor(batch_x4)
            PSNR_u = batch_PSNR(torch.clamp(pre_compressed, 0., 1.), batch_x4_compressed, 1.)
            SSIM_u = batch_SSIM(torch.clamp(pre_compressed, 0., 1.), batch_x4_compressed, 1.)
            total_loss_train2 = L1_loss(pre_compressed, batch_x4_compressed) + 0.5*L1_loss(out2_gradient_map, lable_gradient_map) + 0.2*(1-SSIM_u)
                                

            total_loss_train = 0.5*total_loss_train1 + total_loss_train2
            total_loss_train.backward()
            optimizer_model.step()

            b = time.time()
            epoch_time = b - a
            print_log(epoch, 6000, epoch_time, PSNR1, SSIM1, PSNR_u1, SSIM_u1, PSNR, SSIM, PSNR_u, SSIM_u, total_loss_train, 'model_syn4')
            counter += 1
        if (epoch % 50 == 0):
            save_dict = {'epoch': epoch,
                         'optimizer_state_dict': optimizer_model.state_dict(),
                         'model_state_dict': model.state_dict(),
                         'scheduler': scheduler.state_dict()
                         }
            torch.save(save_dict, os.path.join(args.checkpoint_dir, 'model.pkl'))
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'model' + str(epoch) + '.pkl'))
            # torch.save(model.state_dict(), 'model%d.pkl' % (epoch))
        scheduler.step()