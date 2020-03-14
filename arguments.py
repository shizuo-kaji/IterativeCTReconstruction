import argparse
import numpy as np
import chainer.functions as F
from consts import activation_func,dtypes,uplayer,downlayer,norm_layer,unettype,optim
import os
from datetime import datetime as dt


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=1)
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU IDs (currently, only single-GPU usage is supported')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--root', '-R', default='data',help='Root directory for data')
    parser.add_argument('--planct_dir', '-Rp', default='',help='dir containing planCT images for discriminator')
    parser.add_argument('--mvct_dir', '-Rm', default='',help='dir containing reconstructed MVCT images for discriminator')
    parser.add_argument('--sinogram', '-s', default='', help='directory containing sinograms')
    parser.add_argument('--argfile', '-a', help="specify args file to load settings from")
    parser.add_argument('--projection_matrix', '-pm', default='projection_matrix_2d_256_1074mm.npz',
                        #default='projection_matrix_2d_512_1074mm.npz',
                        help='filename of the projection matrix')
    parser.add_argument('--system_matrix', '-sm', default='systemMatrix_2d_256_1074mm.npz',
                        #default='systemMatrix_2d_1-074mm.npz',
                        help='filename of the system matrix')

    parser.add_argument('--snapinterval', '-si', type=int, default=-1, 
                        help='take snapshot every this reconstruction')
    parser.add_argument('--weight_decay', '-wd', type=float, default=0,
                        help='weight decay for regularization')
    parser.add_argument('--optimizer', '-op',choices=optim.keys(),default='Adam_d',
                        help='select optimizer')
    parser.add_argument('--optimizer_dis', '-opd',choices=optim.keys(),default='Adam_d',
                        help='select optimizer')

    parser.add_argument('--dtype', '-dt', choices=dtypes.keys(), default='fp32',
                        help='floating point precision')
    parser.add_argument('--eqconv', '-eq', action='store_true',
                        help='Enable Equalised Convolution')
    parser.add_argument('--spconv', '-sp', action='store_true',
                        help='Enable Separable Convolution')
    parser.add_argument('--senet', '-se', action='store_true',
                        help='Enable Squeeze-and-Excitation mechanism')

    # data augmentation
    parser.add_argument('--random_translate', '-rt', type=int, default=4, help='random translation for planCT')
    parser.add_argument('--noise_dis', '-nd', type=float, default=0,
                        help='strength of noise injection for discriminator')
    parser.add_argument('--noise_gen', '-ng', type=float, default=0,
                        help='strength of noise injection for generator')

    # discriminator
    parser.add_argument('--dis_activation', '-da', default='lrelu', choices=activation_func.keys())
    parser.add_argument('--dis_out_activation', '-do', default='none', choices=activation_func.keys())
    parser.add_argument('--dis_chs', '-dc', type=int, nargs="*", default=None,
                        help='Number of channels in down layers in discriminator')
    parser.add_argument('--dis_basech', '-db', type=int, default=32,
                        help='the base number of channels in discriminator (doubled in each down-layer)')
    parser.add_argument('--dis_ndown', '-dl', type=int, default=4,
                        help='number of down layers in discriminator')
    parser.add_argument('--dis_ksize', '-dk', type=int, default=4,
                        help='kernel size for patchGAN discriminator')
    parser.add_argument('--dis_down', '-dd', default='down',
                        help='type of down layers in discriminator')
    parser.add_argument('--dis_sample', '-ds', default='down', 
                        help='type of first conv layer for patchGAN discriminator')
    parser.add_argument('--dis_jitter', type=float, default=0,
                        help='jitter for discriminator label for LSGAN')
    parser.add_argument('--dis_dropout', '-ddo', type=float, default=None, 
                        help='dropout ratio for discriminator')
    parser.add_argument('--dis_norm', '-dn', default='instance',
                        choices=norm_layer)
    parser.add_argument('--dis_reg_weighting', '-dw', type=float, default=0,
                        help='regularisation of weighted discriminator. Set 0 to disable weighting')
    parser.add_argument('--dis_wgan', action='store_true',help='WGAN-GP')
    parser.add_argument('--dis_attention', action='store_true',help='attention mechanism for discriminator')

    # generator: G: A -> B, F: B -> A
    parser.add_argument('--gen_activation', '-ga', default='relu', choices=activation_func.keys())
    parser.add_argument('--gen_out_activation', '-go', default='tanh', choices=activation_func.keys())
    parser.add_argument('--gen_fc_activation', '-gfca', default='relu', choices=activation_func.keys())
    parser.add_argument('--gen_chs', '-gc', type=int, nargs="*", default=None,
                        help='Number of channels in down layers in generator')
    parser.add_argument('--gen_ndown', '-gl', type=int, default=3,
                        help='number of down layers in generator')
    parser.add_argument('--gen_basech', '-gb', type=int, default=32,
                        help='the base number of channels in generator (doubled in each down-layer)')
    parser.add_argument('--gen_fc', '-gfc', type=int, default=0,
                        help='number of fc layers before convolutional layers')
    parser.add_argument('--gen_nblock', '-gnb', type=int, default=4,
                        help='number of residual blocks in generators')
    parser.add_argument('--gen_ksize', '-gk', type=int, default=3,    # 4 in the original paper
                        help='kernel size for generator')
    parser.add_argument('--gen_sample', '-gs', default='none-7',
                        help='first and last conv layers for generator')
    parser.add_argument('--gen_down', '-gd', default='down',
                        help='down layers in generator')
    parser.add_argument('--gen_up', '-gu', default='resize',
                        help='up layers in generator')
    parser.add_argument('--gen_dropout', '-gdo', type=float, default=None, 
                        help='dropout ratio for generator')
    parser.add_argument('--gen_norm', '-gn', default='instance',
                        choices=norm_layer)
    parser.add_argument('--unet', '-u', default='none', choices=unettype,
                        help='use u-net skip connections for generator')
    parser.add_argument('--skipdim', '-sd', type=int, default=4,
                        help='channel number for skip connections')
    parser.add_argument('--latent_dim', '-ld', default=-1, type=int,
                        help='dimension of the latent space between encoder and decoder')

    ## loss function
    parser.add_argument('--tv_tau', '-tt', type=float, default=1e-3,
                        help='smoothing parameter for total variation')
    parser.add_argument('--tv_method', '-tm', default='usual', choices=['abs','sobel','usual'],
                        help='method of calculating total variation')

    ##
    parser.add_argument('--epoch', '-e', default=-1, type=int,
                        help='number of reconstructions')
    parser.add_argument('--iter', '-i', default=20000, type=int,
                        help='number of iterations for each reconstruction') 
    parser.add_argument('--vis_freq', '-vf', default=2000, type=int,
                        help='image output interval')

    parser.add_argument('--max_reconst_freq', '-mf', default=1, type=int,   # 40
                        help='consistency loss will be considered one in every this number in the end')
    parser.add_argument('--reconst_freq_decay_start', '-rfd', default=400, type=int,
                        help='reconst_freq starts to increase towards max_reconst_freq after this number of iterations')
    parser.add_argument('--dis_freq', '-df', default=-1, type=int,
                        help='discriminator update interval; set to negative to turn of discriminator')

    parser.add_argument('--lr_sd', '-lrs', default=1e-2, type=float,   # 1e-2 for exp, 0.5 for log, 1e-1 for conjugate
                        help='learning rate for seed array')
    parser.add_argument('--lr_gen', '-lrg', default=2e-4, type=float, # 1e-2
                        help='learning rate for generator NN')
    parser.add_argument('--lr_dis', '-lrd', default=1e-4, type=float,
                        help='learning rate for discriminator NN')
    parser.add_argument('--lr_drop', '-lrp', default=1, type=int,
                        help='learning rate decay')
    parser.add_argument('--no_train_dec', '-ntd', action='store_true', help='not updating decoder during training')
    parser.add_argument('--no_train_enc', '-nte', action='store_true', help='not updating encoder during training')
    parser.add_argument('--no_train_seed', '-nts', action='store_true', help='not updating seed during training')
    parser.add_argument('--decoder_only', '-d', action='store_true', help='not using encoder')
    parser.add_argument('--clip', '-cl', action='store_true')

    parser.add_argument('--lambda_tv', '-ltv', default=0, type=float,   # 2e+2 for 256x256, 5e+2 is strong
                        help='weight of total variation regularization for generator')
    parser.add_argument('--lambda_tvs', '-ltvs', default=0, type=float,   # 
                        help='weight of total variation regularization for seed array')
    parser.add_argument('--lambda_adv', '-ladv', default=0.0, type=float, 
                        help='weight of adversarial loss for generator')
    parser.add_argument('--lambda_advs', '-ladvs', default=0.0, type=float, 
                        help='weight of adversarial loss for seed')
    parser.add_argument('--lambda_gan', '-lgan', default=0.0, type=float,
                        help='weight of random fake generation loss for generator')
    parser.add_argument('--lambda_sd', '-ls', default=0, type=float,
                        help='weight of reconstruction consistency loss for seed array')
    parser.add_argument('--lambda_nn', '-ln', default=1.0, type=float, 
                        help='weight of reconstruction consistency loss for CNN')
    parser.add_argument('--lambda_ae1', '-lae1', default=0.0, type=float, 
                        help='autoencoder L1 loss for generator')
    parser.add_argument('--lambda_ae2', '-lae2', default=0.0, type=float, 
                        help='autoencoder L2 loss for generator')
    parser.add_argument('--lambda_reg', '-lreg', type=float, default=0,
                        help='weight for regularisation for generator')

    parser.add_argument('--model_gen', '-mg', help='pretrained model file for generator')
    parser.add_argument('--model_dis', '-md', help='pretrained model file for discriminator')
    parser.add_argument('--model_image', '-mi', default="", help='initial seed image')
    parser.add_argument('--crop_width', '-cw', type=int, default=256) 
    parser.add_argument('--crop_height', '-ch', type=int, default=None)
    parser.add_argument('--scale_to', '-sc', type=int, default=-1)

    ## dicom related
    parser.add_argument('--HU_base', '-hub', type=int, default=-4500,     # -4500, 
                        help='minimum HU value to be accounted for')
    parser.add_argument('--HU_range', '-hur', type=int, default=6000, # 6000,
                        help='the maximum HU value to be accounted for will be HU_base+HU_range') #700
    parser.add_argument('--HU_range_vis', '-hurv', default=2000, type=int,
                        help='HU range in the visualization')

    args = parser.parse_args()
    if not args.gen_chs:
        args.gen_chs = [int(args.gen_basech) * (2**i) for i in range(args.gen_ndown)]
    if not args.dis_chs:
        args.dis_chs = [int(args.dis_basech) * (2**i) for i in range(args.dis_ndown)]
    if not args.planct_dir:
        args.planct_dir = os.path.join(args.root,"planCT")
    if not args.mvct_dir:
        args.mvct_dir = os.path.join(args.root,"reconstructed")
    if not args.sinogram:
        args.sinogram = os.path.join(args.root,"projection")
    if not args.crop_height:
        args.crop_height = args.crop_width
    if args.latent_dim>0:
        args.decoder_only = True
    args.ch = 1
    args.out_ch = 1
    dtime = dt.now().strftime('%m%d_%H%M')
    args.out = os.path.join(args.out, '{}_ln{}_lgan{}_ladv{}_df{},dim{}_mg{}_md{}'.format(dtime,args.lambda_nn,args.lambda_gan,args.lambda_adv,args.dis_freq,args.latent_dim,(args.model_gen is not None),(args.model_dis is not None)))
    return(args)

