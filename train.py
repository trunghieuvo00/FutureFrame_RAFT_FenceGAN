import os
from glob import glob
import cv2
import time
import datetime
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import argparse
import random

from utils import *
from losses import *
import Dataset
from models.unet import UNet
#from models.models_nestedUnet import NestedUNet
#from models.UNet3.UNet_3Plus import UNet_3Plus, UNet_3Plus_DeepSup
from models.R2U_Net.r2unet import R2U_Net, RecU_Net, ResU_Net
from models.pix2pix_networks import PixelDiscriminator
from models.liteFlownet import lite_flownet as lite_flow
from config import update_config
from models.flownet2.models import FlowNet2SD
from evaluate import val

from models.RAFT.models_raft import RAFT

from models.gmflow.gmflow.gmflow import GMFlow

# Usage GPU
import os

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

parser = argparse.ArgumentParser(description='Anomaly Prediction')
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--dataset', default='avenue', type=str, help='The name of the dataset to train.')
parser.add_argument('--iters', default=40000, type=int, help='The total iteration number.')
parser.add_argument('--resume', default=None, type=str,
                    help='The pre-trained model to resume training with, pass \'latest\' or the model name.')
parser.add_argument('--save_interval', default=1000, type=int, help='Save the model every [save_interval] iterations.')
parser.add_argument('--val_interval', default=1000, type=int,
                    help='Evaluate the model every [val_interval] iterations, pass -1 to disable.')
parser.add_argument('--show_flow', default=False, action='store_true',
                    help='If True, the first batch of ground truth optic flow could be visualized and saved.')
parser.add_argument('--flownet', default='raft', type=str, help='RAFT standard model')
## Fence GAN
parser.add_argument('--fence_gamma', default=0.1, type=str, help='Gamma coefficient in Discriminator loss')
parser.add_argument('--fence_alpha', default=0.5, type=str, help='Gamma coefficient in Discriminator loss')
parser.add_argument('--fence_beta', default=15, type=str, help='Gamma coefficient in Discriminator loss')


### RAFT arguments

#parser.add_argument('--name', default='raft', help="name your experiment")
#parser.add_argument('--stage', help="determines which dataset to use for training") 
#parser.add_argument('--restore_ckpt', help="restore checkpoint")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--validation', type=str, nargs='+')

parser.add_argument('--lr', type=float, default=0.00002)
parser.add_argument('--num_steps', type=int, default=100000)
parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
#parser.add_argument('--gpus', type=int, nargs='+', default=1)
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

parser.add_argument('--wdecay', type=float, default=.00005)
parser.add_argument('--epsilon', type=float, default=1e-8)
parser.add_argument('--clip', type=float, default=1.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
parser.add_argument('--add_noise', action='store_true')

parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--path', help="dataset for evaluation")
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')


# GMFlow model
'''
parser.add_argument('--num_scales', default=1, type=int,
                    help='basic gmflow model uses a single 1/8 feature, the refinement uses 1/4 feature')
parser.add_argument('--feature_channels', default=128, type=int)
parser.add_argument('--upsample_factor', default=8, type=int)
parser.add_argument('--num_transformer_layers', default=6, type=int)
parser.add_argument('--num_head', default=1, type=int)
parser.add_argument('--attention_type', default='swin', type=str)
parser.add_argument('--ffn_dim_expansion', default=4, type=int)

parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                    help='number of splits in attention')
parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                    help='correlation radius for matching, -1 indicates global matching')
parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                    help='self-attention radius for flow propagation, -1 indicates global attention')
                    
'''

args = parser.parse_args()
train_cfg = update_config(args, mode='train')
train_cfg.print_cfg()

generator = UNet(input_channels=12, output_channel=3).cuda()
#generator = UNet_3Plus(in_channels=12, out_channels=3).cuda()
#generator = NestedUNet(in_channels=12, out_channels=3).cuda()
#generator = R2U_Net(img_ch=12, output_ch=3, t=3).cuda()
#generator = ResU_Net(img_ch=12, output_ch=3).cuda()
#generator = RecU_Net(img_ch=12, output_ch=3, t=3).cuda()


discriminator = PixelDiscriminator(input_nc=3).cuda()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=train_cfg.g_lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=train_cfg.d_lr)

if train_cfg.resume:
    generator.load_state_dict(torch.load(train_cfg.resume)['net_g'])
    discriminator.load_state_dict(torch.load(train_cfg.resume)['net_d'])
    optimizer_G.load_state_dict(torch.load(train_cfg.resume)['optimizer_g'])
    optimizer_D.load_state_dict(torch.load(train_cfg.resume)['optimizer_d'])
    print(f'Pre-trained generator and discriminator have been loaded.\n')
else:
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    print('Generator and discriminator are going to be trained from scratch.\n')

assert train_cfg.flownet in ('2sd', 'raft'), 'Flow net only supports LiteFlownet or FlowNet2SD currently.'
if train_cfg.flownet == '2sd':
    flow_net = FlowNet2SD()
    flow_net.load_state_dict(torch.load('models/flownet2/FlowNet2-SD.pth')['state_dict'])
else:
    flow_net = RAFT(args)
    '''flow_net = GMFlow(feature_channels=args.feature_channels,
               num_scales=args.num_scales,
               upsample_factor=args.upsample_factor,
               num_head=args.num_head,
               attention_type=args.attention_type,
               ffn_dim_expansion=args.ffn_dim_expansion,
               num_transformer_layers=args.num_transformer_layers,
               ).to(device)'''
    flow_net.load_state_dict(torch.load('models/RAFT/pretrained/raft-sintel.pth'), strict=False)
    
    #flow_net = flow_net.module

flow_net.cuda().eval()  # Use flow_net to generate optic flows, so set to eval mode.

adversarial_loss = Adversarial_Loss().cuda()
#discriminate_loss = Discriminate_Loss().cuda()
discriminate_loss = Discriminate_Loss_Fence().cuda()
gradient_loss = Gradient_Loss(3).cuda()
flow_loss = Flow_Loss().cuda()
intensity_loss = Intensity_Loss().cuda()
dispersion_loss = Dispersion_Loss().cuda()

train_dataset = Dataset.train_dataset(train_cfg)

# Remember to set drop_last=True, because we need to use 4 frames to predict one frame.
train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)

writer = SummaryWriter(f'tensorboard_log/{train_cfg.dataset}_bs{train_cfg.batch_size}')
start_iter = int(train_cfg.resume.split('_')[-1].split('.')[0]) if train_cfg.resume else 0
training = True
generator = generator.train()
discriminator = discriminator.train()

try:
    step = start_iter
    while training:
        for indice, clips, flow_strs in train_dataloader:
            input_frames = clips[:, 0:12, :, :].cuda()  # (n, 12, 256, 256) ==> Explain: We have 3 channel for each frame so there're 4 input frame that equivalent 12
            input_last = input_frames[:, 9:12, :, :].cuda()  # use for flow_loss
            #input_frames = (input_frames + torch.randn(input_frames.size())).cuda()
            
            target_frame = clips[:, 12:15, :, :].cuda()  # (n, 3, 256, 256)

            # pop() the used frame index, this can't work in train_dataset.__getitem__ because of multiprocessing.
            for index in indice:
                train_dataset.all_seqs[index].pop()
                if len(train_dataset.all_seqs[index]) == 0:
                    train_dataset.all_seqs[index] = list(range(len(train_dataset.videos[index]) - 4))
                    random.shuffle(train_dataset.all_seqs[index])

            G_frame = generator(input_frames)

            if train_cfg.flownet == 'raft':
                #gt_flow_input = torch.cat([input_last, target_frame], 1)
                #pred_flow_input = torch.cat([input_last, G_frame], 1)
                ## No need to train flow_net, use .detach() to cut off gradients.
                #flow_gt = flow_net.forward(gt_flow_input, flow_net).detach()
                #flow_pred = flow_net.batch_estimate(pred_flow_input, flow_net).detach()
                #******
                #gt_flow_input = torch.cat([input_last, target_frame], 1)
                #pred_flow_input = torch.cat([input_last, G_frame], 1)
                ## No need to train flow_net, use .detach() to cut off gradients.
                t, flow_gt = flow_net.forward(input_last, target_frame, iters=20, test_mode=True)
                t, flow_pred = flow_net.forward(input_last, G_frame, iters=20, test_mode=True)
            else:
                gt_flow_input = torch.cat([input_last.unsqueeze(2), target_frame.unsqueeze(2)], 2)
                pred_flow_input = torch.cat([input_last.unsqueeze(2), G_frame.unsqueeze(2)], 2)

                flow_gt = (flow_net(gt_flow_input * 255.) / 255.).detach()  # Input for flownet2sd is in (0, 255).
                flow_pred = (flow_net(pred_flow_input * 255.) / 255.).detach()

            if train_cfg.show_flow:
                flow = np.array(flow_gt.cpu().detach().numpy().transpose(0, 2, 3, 1), np.float32)  # to (n, w, h, 2)
                for i in range(flow.shape[0]):
                    aa = flow_to_color(flow[i], convert_to_bgr=False)
                    path = train_cfg.train_data.split('/')[-3] + '_' + flow_strs[i]
                    cv2.imwrite(f'images/{path}.jpg', aa)  # e.g. images/avenue_4_574-575.jpg
                    print(f'Saved a sample optic flow image from gt frames: \'images/{path}.jpg\'.')

            inte_l = intensity_loss(G_frame, target_frame)
            grad_l = gradient_loss(G_frame, target_frame)
            fl_l = flow_loss(flow_pred, flow_gt)
            g_l = adversarial_loss(discriminator(G_frame))
            #FenceGAN
            loss_dispersion = dispersion_loss(G_out=G_frame)
            
            #G_l_t = 1. * inte_l + 1. * grad_l + 2. * fl_l + 0.05 * g_l
            ## TODO:  (1. * inte_l + 1. * grad_l + 2. * fl_l + g_l) /2
            #G_l_t = 1. * inte_l + 1. * grad_l + 2. * fl_l + 0.05 * (g_l + 15*loss_dispersion) 
            G_l_t = 1. * inte_l + 1. * grad_l + 2. * fl_l + 0.05 * g_l

            # When training discriminator, don't train generator, so use .detach() to cut off gradients.
            D_l = discriminate_loss(discriminator(target_frame), discriminator(G_frame.detach()))

            

            # Or just do .step() after all the gradients have been computed, like the following way:
            optimizer_D.zero_grad()
            D_l.backward()
            optimizer_G.zero_grad()
            G_l_t.backward()
            optimizer_D.step()
            optimizer_G.step()

            torch.cuda.synchronize()
            time_end = time.time()
            if step > start_iter:  # This doesn't include the testing time during training.
                iter_t = time_end - temp
            temp = time_end

            if step != start_iter:
                if step % 20 == 0:
                    time_remain = (train_cfg.iters - step) * iter_t
                    eta = str(datetime.timedelta(seconds=time_remain)).split('.')[0]
                    psnr = psnr_error(G_frame, target_frame)
                    lr_g = optimizer_G.param_groups[0]['lr']
                    lr_d = optimizer_D.param_groups[0]['lr']

                    print(f"[{step}]  inte_l: {inte_l:.3f} | grad_l: {grad_l:.3f} | fl_l: {fl_l:.3f} | "
                          f"g_l: {g_l:.3f} | G_l_total: {G_l_t:.3f} | D_l: {D_l:.3f} | psnr: {psnr:.3f} | "
                          f"iter: {iter_t:.3f}s | ETA: {eta} | lr: {lr_g} {lr_d}")

                    save_G_frame = ((G_frame[0] + 1) / 2)
                    save_G_frame = save_G_frame.cpu().detach()[(2, 1, 0), ...]
                    save_target = ((target_frame[0] + 1) / 2)
                    save_target = save_target.cpu().detach()[(2, 1, 0), ...]

                    writer.add_scalar('psnr/train_psnr', psnr, global_step=step)
                    writer.add_scalar('total_loss/g_loss_total', G_l_t, global_step=step)
                    writer.add_scalar('total_loss/d_loss', D_l, global_step=step)
                    writer.add_scalar('G_loss_total/g_loss', g_l, global_step=step)
                    writer.add_scalar('G_loss_total/fl_loss', fl_l, global_step=step)
                    writer.add_scalar('G_loss_total/inte_loss', inte_l, global_step=step)
                    writer.add_scalar('G_loss_total/grad_loss', grad_l, global_step=step)
                    writer.add_scalar('psnr/train_psnr', psnr, global_step=step)

                if step % int(train_cfg.iters / 100) == 0:
                    writer.add_image('image/G_frame', save_G_frame, global_step=step)
                    writer.add_image('image/target', save_target, global_step=step)

                if step % train_cfg.save_interval == 0:
                    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                                  'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                    torch.save(model_dict, f'weights/{train_cfg.dataset}_{step}.pth')
                    print(f'\nAlready saved: \'{train_cfg.dataset}_{step}.pth\'.')

                if step % train_cfg.val_interval == 0:
                    auc = val(train_cfg, model=generator)
                    writer.add_scalar('results/auc', auc, global_step=step)
                    generator.train()

            step += 1
            if step > train_cfg.iters:
                training = False
                model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                              'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
                torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{step}.pth')
                break
                

except KeyboardInterrupt:
    print(f'\nStop early, model saved: \'latest_{train_cfg.dataset}_{step}.pth\'.\n')

    if glob(f'weights/latest*'):
        os.remove(glob(f'weights/latest*')[0])

    model_dict = {'net_g': generator.state_dict(), 'optimizer_g': optimizer_G.state_dict(),
                  'net_d': discriminator.state_dict(), 'optimizer_d': optimizer_D.state_dict()}
    torch.save(model_dict, f'weights/latest_{train_cfg.dataset}_{step}.pth')
