##########################
# Scaling convention
# NN => HU: NN/2 * range + base
# RAW => HU: 1024 * ( RAW - 0.0716 )/ 0.0716
# HU => RAW: HU * 0.0716 / 1024 + 0.0716

import argparse
import os
import sys

import numpy as np
import random,math
import scipy
import cupyx

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable, optimizers, serializers, training
from chainer.training import extensions
from chainerui.extensions import CommandsExtension
from chainerui.utils import save_args

import cupy as cp

from dataset import Dataset,prjData 

from net import Discriminator
import losses
from arguments import arguments
from consts import dtypes,optim
from updater import Updater

#import scanconf

def plot_ylimit(f,a,summary):
    a.set_ylim(top=0.03)
def plot_log(f,a,summary):
    a.set_yscale('log')

#########################
def main():
    args = arguments()
    chainer.config.autotune = True
    chainer.print_runtime_info()
    print(args)

    if args.dp:
        from net_dp import Encoder,Decoder
    else:
        from net import Encoder,Decoder

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        xp = cuda.cupy
        sp = cupyx.scipy.sparse
    else:
        print("runs desperately slowly without a GPU!")
        xp = np
        sp = scipy.sparse

    ##  Input information ##
#    InputFile = scanconf.ScanConfig()
#    InputFile.reconSize = args.crop_width
    
    ## setup trainable links
    decoder = Decoder(args)
    if args.use_enc:
        encoder = Encoder(args)
    else:
        encoder = L.Linear(1)
    if args.use_dis:
        dis = Discriminator(args)
    else:
        dis = L.Linear(1)

    if args.model_dis:
        serializers.load_npz(args.model_dis, dis) 
        print('discriminator model loaded: {}'.format(args.model_dis))
    if args.model_gen:
        if 'enc' in args.model_gen and not args.decoder_only:
            serializers.load_npz(args.model_gen, encoder)
            print('encoder model loaded: {}'.format(args.model_gen))
        serializers.load_npz(args.model_gen.replace('enc','dec'), decoder) 
        print('decoder model loaded: {}'.format(args.model_gen.replace('enc','dec')))
    if args.lambda_sd>0 and args.lr_sd < 0.05:
        print("\n\n for usual iterative reconstruction (-ls), --lr_sd should be around 0.1. \n\n")
    if args.latent_dim>0:
        init = xp.zeros((args.batchsize,args.latent_dim)).astype(np.float32)
    elif args.decoder_only:
        init = xp.zeros((args.batchsize,decoder.latent_c,decoder.latent_h,decoder.latent_w)).astype(np.float32)
    else:
        init = xp.zeros((args.batchsize,1,args.crop_height,args.crop_width)).astype(np.float32)
#    init = xp.random.uniform(-0.1,0.1,(1,1,args.crop_height,args.crop_width)).astype(np.float32)
    print("Initial image {} shape {}".format(args.model_image,init.shape))
    seed = L.Parameter(init)

    if args.gpu>=0:
        decoder.to_gpu()
        seed.to_gpu()
        encoder.to_gpu()
        dis.to_gpu()

    # setup optimisers
    def make_optimizer(model, lr, opttype='Adam'):
#        eps = 1e-5 if args.dtype==np.float16 else 1e-8
        optimizer = optim[opttype](lr)
        #from profiled_optimizer import create_marked_profile_optimizer
#        optimizer = create_marked_profile_optimizer(optim[opttype](lr), sync=True, sync_level=2)
        optimizer.setup(model)
        if args.weight_decay>0:
            if opttype in ['Adam','Adam_d','AdaBound','Eve']:
                optimizer.weight_decay_rate = args.weight_decay
            else:
                optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(args.weight_decay))
#        optimizer.add_hook(chainer.optimizer_hooks.GradientClipping(100))
        return optimizer

    optimizer_sd = make_optimizer(seed, args.lr_sd, args.optimizer)
    optimizer_dec = make_optimizer(decoder, args.lr_gen, args.optimizer)
    optimizer_enc = make_optimizer(encoder, args.lr_gen, args.optimizer)
    optimizer_dis = make_optimizer(dis, args.lr_dis, args.optimizer_dis)

#  unify CPU and GPU memory to load big matrices
#    if args.crop_height>256:
#        pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
#        cp.cuda.set_allocator(pool.malloc)

    # projection matrices
    prMat, conjMat = None, None
    if args.lambda_sd > 0 or args.lambda_nn > 0:
        prMat = scipy.sparse.load_npz(os.path.join(args.root,args.projection_matrix))
#        cx = prMat.tocsr()
#        rows,cols = cx.nonzero()
#        for i,j in zip(rows,cols):
#            if cx[i,j] < 1e-5:
#                cx[i,j] = 0
#        prMat = cx.tocoo()
#        scipy.sparse.save_npz("d:/ml/reconst/pr.npz",prMat)
        prMat = sp.coo_matrix(prMat, dtype = np.float32)
        prMat = chainer.utils.CooMatrix(prMat.data, prMat.row, prMat.col, prMat.shape)
        print("Projection matrix {} shape {}".format(args.projection_matrix,prMat.shape,))
        if args.system_matrix:
            conjMat = scipy.sparse.load_npz(os.path.join(args.root,args.system_matrix))
            conjMat = sp.coo_matrix(conjMat, dtype = np.float32)
            conjMat = chainer.utils.CooMatrix(conjMat.data, conjMat.row, conjMat.col, conjMat.shape)
            print("Conjugate matrix {} shape {}".format(args.system_matrix,conjMat.shape))

    # setup updater
    print("Setting up data iterators...")
    planct_dataset = Dataset(
        path=args.planct_dir, baseA=args.HU_base, rangeA=args.HU_range, crop=(args.crop_height,args.crop_width),
        scale_to=args.scale_to, random=args.random_translate) 
    planct_iter = chainer.iterators.SerialIterator(planct_dataset, args.batchsize, shuffle=True)
    mvct_dataset = Dataset(
        path=args.mvct_dir, baseA=args.HU_base, rangeA=args.HU_range, crop=(args.crop_height,args.crop_width),
        scale_to=args.scale_to, random=args.random_translate) 
    mvct_iter = chainer.iterators.SerialIterator(mvct_dataset, args.batchsize, shuffle=True)
    data = prjData(args.sinogram)
    proj_iter = chainer.iterators.SerialIterator(data, args.batchsize, shuffle=False) # True

    updater = Updater(
        models=(seed,encoder,decoder,dis),
        iterator={'main':proj_iter, 'planct':planct_iter, 'mvct':mvct_iter},
        optimizer={'main': optimizer_sd, 'enc': optimizer_enc, 'dec': optimizer_dec, 'dis': optimizer_dis},
        device=args.gpu,
        params={'args': args, 'prMat':prMat, 'conjMat':conjMat}
        )

    # logging
    if args.epoch < 0:
        total_iter = -args.epoch*args.iter * math.ceil(len(data)/args.batchsize)
    else:
        total_iter = args.epoch*args.iter
    trainer = training.Trainer(updater, (total_iter, 'iteration'), out=args.out)
    log_interval = (50, 'iteration')
    log_keys_main = []
    log_keys_dis = []
    log_keys_grad = ['main/grad_sd','main/grad_gen','main/grad_sd_consistency','main/grad_gen_consistency','main/seed_diff']
    loss_main_list = [(args.lambda_sd,'main/loss_sd'),(args.lambda_nn,'main/loss_nn'),(args.lambda_ae1,'main/loss_ae1'),(args.lambda_ae2,'main/loss_ae2'),
                        (args.lambda_tv,'main/loss_tv'),(args.lambda_tvs,'main/loss_tvs'),(args.lambda_reg,'main/loss_reg'),(args.lambda_reg,'main/loss_reg_ae')]
    for a,k in loss_main_list:
        if a>0:
            log_keys_main.append(k)
    loss_dis_list = [(args.lambda_adv,'main/loss_adv'),(args.lambda_advs,'main/loss_advs'),(args.dis_freq,'main/loss_dis'),(args.lambda_gan,'main/loss_gan')]
    for a,k in loss_dis_list:
        if a>0:
            log_keys_dis.append(k)    
    log_keys = ['iteration']+log_keys_main+log_keys_dis+log_keys_grad
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.LogReport(keys=log_keys, trigger=log_interval))
    trainer.extend(extensions.PrintReport(log_keys), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
                log_keys_main, 'iteration',trigger=(100, 'iteration'), file_name='loss.png', postprocess=plot_log))
        trainer.extend(extensions.PlotReport(
                log_keys_dis, 'iteration',trigger=(100, 'iteration'), file_name='loss_dis.png'))
        trainer.extend(extensions.PlotReport(
                log_keys_grad, 'iteration',trigger=(100, 'iteration'), file_name='loss_grad.png', postprocess=plot_log))

#    trainer.extend(extensions.ParameterStatistics([seed,decoder]))   ## very slow
    trainer.extend(CommandsExtension())

    if args.snapinterval <= 0:
        args.snapinterval = total_iter
        
    if args.lambda_nn>0:
        trainer.extend(extensions.dump_graph('main/loss_nn', out_name='gen.dot'))
    elif args.lambda_ae1>0:
        trainer.extend(extensions.dump_graph('main/loss_ae1', out_name='gen.dot'))

    # save models
    if args.use_enc:
        trainer.extend(extensions.snapshot_object(
            encoder, 'enc_{.updater.iteration}.npz'), trigger=(args.snapinterval, 'iteration'))
        trainer.extend(extensions.snapshot_object(
            optimizer_enc, 'opt_enc_{.updater.iteration}.npz'), trigger=(args.snapinterval, 'iteration'))
    if args.use_dis:
        trainer.extend(extensions.snapshot_object(
            dis, 'dis_{.updater.iteration}.npz'), trigger=(args.snapinterval, 'iteration'))
        trainer.extend(extensions.snapshot_object(
            optimizer_dis, 'opt_dis_{.updater.iteration}.npz'), trigger=(args.snapinterval, 'iteration'))
#        trainer.extend(extensions.dump_graph('main/loss_real', out_name='dis.dot'))
    trainer.extend(extensions.snapshot_object(
        decoder, 'dec_{.updater.iteration}.npz'), trigger=(args.snapinterval, 'iteration'))
    trainer.extend(extensions.snapshot_object(
        optimizer_dec, 'opt_dec_{.updater.iteration}.npz'), trigger=(args.snapinterval, 'iteration'))

    # save command line arguments
    os.makedirs(args.out, exist_ok=True)
    save_args(args,args.out)
    with open(os.path.join(args.out,"args.txt"), 'w') as fh:
        fh.write(" ".join(sys.argv))

    trainer.run()
                
if __name__ == '__main__':
    main()
