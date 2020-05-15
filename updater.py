import random
import chainer
import chainer.functions as F
from chainer import Variable,cuda,initializers
import losses
import numpy as np
from chainercv.utils import read_image,write_image
import os
from dataset import write_dicom
import pydicom as dicom

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.seed, self.encoder, self.decoder, self.dis = kwargs.pop('models')
        params = kwargs.pop('params')
        self.args = params['args']
        self.prMat = params['prMat']   ## projection matrix
        self.conjMat = params['conjMat']   ## system matrix
        self._buffer = losses.ImagePool(200)
        self.n_reconst = 0
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        optimizer_sd = self.get_optimizer('main')
        optimizer_enc = self.get_optimizer('enc')
        optimizer_dec = self.get_optimizer('dec')
        optimizer_dis = self.get_optimizer('dis')
        xp = self.seed.xp

        step = self.iteration % self.args.iter
        if step == 0:
            batch = self.get_iterator('main').next()
            self.prImg, self.rev, self.patient_id, self.slice = self.converter(batch, self.device)
            self.n_reconst += 1
            self.recon_freq = 1
            if ".npy" in self.args.model_image:
                self.seed.W.array = xp.reshape(xp.load(self.args.model_image),(1,1,self.args.crop_height,self.args.crop_width))
            elif ".dcm" in self.args.model_image:
                ref_dicom = dicom.read_file(self.args.model_image, force=True)
                img = xp.array(ref_dicom.pixel_array+ref_dicom.RescaleIntercept)
                img = (2*(xp.clip(img,self.args.HU_base,self.args.HU_base+self.args.HU_range)-self.args.HU_base)/self.args.HU_range-1.0).astype(np.float32)
                self.seed.W.array = xp.reshape(img,(1,1,self.args.crop_height,self.args.crop_width))
            else:
#                initializers.Uniform()(self.seed.W.array)
                initializers.HeNormal()(self.seed.W.array)
            self.initial_seed = self.seed.W.array.copy()
#            print(xp.min(self.initial_seed),xp.max(self.initial_seed),xp.mean(self.initial_seed))

        ## for seed array
        arr = self.seed()
        HU = ((arr+1)/2 * self.args.HU_range)+self.args.HU_base  # [-1000=air,0=water,>1000=bone]
        raw = HU * 0.0716 / 1024 + 0.0716

        self.seed.cleargrads()
        loss_seed = Variable(xp.array([0.0],dtype=np.float32))
        # conjugate correction using system matrix
        if self.args.lambda_sd > 0:
            self.seed.W.grad = xp.zeros_like(self.seed.W.array)
            loss_sd = 0
            for i in range(len(self.prImg)):
                if self.rev[i]:
                    rec_sd = F.exp(-F.sparse_matmul(self.prMat,F.reshape(raw[i,:,::-1,::-1],(-1,1)))) ##
                else:
                    rec_sd = F.exp(-F.sparse_matmul(self.prMat,F.reshape(raw[i],(-1,1)))) ##
                loss_sd += F.mean_squared_error(rec_sd,self.prImg[i])
                if self.args.system_matrix:
                    gd = F.sparse_matmul( self.conjMat, rec_sd-self.prImg[i], transa=True)
                    if self.rev[i]:
                        self.seed.W.grad[i] -= self.args.lambda_sd * F.reshape(gd, (1,self.args.crop_height,self.args.crop_width)).array[:,::-1,::-1]    # / logrep.shape[0] ?
                    else:
                        self.seed.W.grad[i] -= self.args.lambda_sd * F.reshape(gd, (1,self.args.crop_height,self.args.crop_width)).array    # / logrep.shape[0] ?

            if not self.args.system_matrix:
                (self.args.lambda_sd *loss_sd).backward()
            chainer.report({'loss_sd': loss_sd/len(self.prImg)}, self.seed)

        if self.args.lambda_tvs > 0:
            loss_tvs = losses.total_variation(arr, tau=self.args.tv_tau, method=self.args.tv_method)
            loss_seed += self.args.lambda_tvs * loss_tvs
            chainer.report({'loss_tvs': loss_tvs}, self.seed)

        if self.args.lambda_advs>0:
            L_advs = F.average( (self.dis(arr)-1.0)**2 )
            loss_seed += self.args.lambda_advs * L_advs
            chainer.report({'loss_advs': L_advs}, self.seed)

        ## generator output
        arr_n = losses.add_noise(arr,self.args.noise_gen)
        if self.args.no_train_seed:
            arr_n.unchain()
        if not self.args.decoder_only:
            arr_n = self.encoder(arr_n)
        gen = self.decoder(arr_n) # range = [-1,1]

        ## generator loss
        loss_gen = Variable(xp.array([0.0],dtype=np.float32))
        plan, plan_ae = None, None
        if self.args.lambda_ae1>0 or self.args.lambda_ae2>0:
            plan = losses.add_noise(Variable(self.converter(self.get_iterator('planct').next(), self.device)), self.args.noise_dis)
            plan_enc = self.encoder(plan)
            plan_ae = self.decoder(plan_enc)
            loss_ae1 = F.mean_absolute_error(plan,plan_ae)
            loss_ae2 = F.mean_squared_error(plan,plan_ae)
            if self.args.lambda_reg>0:
                loss_reg_ae = losses.loss_func_reg(plan_enc[-1],'l2')
                chainer.report({'loss_reg_ae': loss_reg_ae}, self.seed)
                loss_gen += self.args.lambda_reg * loss_reg_ae
            loss_gen += self.args.lambda_ae1 * loss_ae1 + self.args.lambda_ae2 * loss_ae2
            chainer.report({'loss_ae1': loss_ae1}, self.seed)
            chainer.report({'loss_ae2': loss_ae2}, self.seed)
        if self.args.lambda_tv > 0:
            L_tv = losses.total_variation(gen, tau=self.args.tv_tau, method=self.args.tv_method)
            loss_gen += self.args.lambda_tv * L_tv
            chainer.report({'loss_tv': L_tv}, self.seed)
        if self.args.lambda_adv>0:
            L_adv = F.average( (self.dis(gen)-1.0)**2 )
            loss_gen += self.args.lambda_adv * L_adv
            chainer.report({'loss_adv': L_adv}, self.seed)
        ## regularisation on the latent space
        if self.args.lambda_reg>0:
            loss_reg = losses.loss_func_reg(arr_n[-1],'l2')
            chainer.report({'loss_reg': loss_reg}, self.seed)
            loss_gen += self.args.lambda_reg * loss_reg

        self.encoder.cleargrads()
        self.decoder.cleargrads()
        loss_gen.backward()
        loss_seed.backward()
        chainer.report({'loss_gen': loss_gen}, self.seed)
        optimizer_enc.update()
        optimizer_dec.update()
        optimizer_sd.update()

        chainer.report({'grad_sd': F.average(F.absolute(self.seed.W.grad))}, self.seed)
        if hasattr(self.decoder, 'latent_fc'):
            chainer.report({'grad_gen': F.average(F.absolute(self.decoder.latent_fc.W.grad))}, self.seed)

        # reconstruction consistency for NN
        if (step % self.recon_freq == 0) and self.args.lambda_nn>0:
            self.encoder.cleargrads()
            self.decoder.cleargrads()
            self.seed.cleargrads()
            gen.grad = xp.zeros_like(gen.array)

            HU_nn = ((gen+1)/2 * self.args.HU_range)+self.args.HU_base  # [-1000=air,0=water,>1000=bone]
            raw_nn = HU_nn * 0.0716 / 1024 + 0.0716
            loss_nn = 0
            for i in range(len(self.prImg)):
                if self.rev[i]:
                    rec_nn = F.exp(-F.sparse_matmul(self.prMat,F.reshape(raw_nn[i,:,::-1,::-1],(-1,1))))
                else:
                    rec_nn = F.exp(-F.sparse_matmul(self.prMat,F.reshape(raw_nn[i],(-1,1))))
                loss_nn += F.mean_squared_error(rec_nn,self.prImg[i])
                if self.args.system_matrix:
                    gd_nn = F.sparse_matmul( rec_nn-self.prImg[i], self.conjMat, transa=True )
                    if self.rev[i]:
                        gen.grad[i] -= self.args.lambda_nn * F.reshape(gd_nn, (1,self.args.crop_height,self.args.crop_width)).array[:,::-1,::-1]
                    else:
                        gen.grad[i] -= self.args.lambda_nn * F.reshape(gd_nn, (1,self.args.crop_height,self.args.crop_width)).array
            chainer.report({'loss_nn': loss_nn/len(self.prImg)}, self.seed)
            if self.args.system_matrix:
                gen.backward()
            else:
                (self.args.lambda_nn * loss_nn).backward()

            if not self.args.no_train_seed:
                optimizer_sd.update()
            if not self.args.no_train_enc:
                optimizer_enc.update()
            if not self.args.no_train_dec:
                optimizer_dec.update()

            if self.seed.W.grad is not None:
                chainer.report({'grad_sd_consistency': F.average(F.absolute(self.seed.W.grad))}, self.seed)
            if hasattr(self.decoder, 'latent_fc'):
                chainer.report({'grad_gen_consistency': F.average(F.absolute(self.decoder.latent_fc.W.grad))}, self.seed)
            elif hasattr(self.decoder, 'ul'):
                chainer.report({'grad_gen_consistency': F.average(F.absolute(self.decoder.ul.c1.c.W.grad))}, self.seed)

        chainer.report({'seed_diff': F.mean_absolute_error(self.initial_seed,self.seed.W)/F.mean_absolute_error(self.initial_seed,xp.zeros_like(self.initial_seed))}, self.seed)

        # clip seed to [-1,1]
        if self.args.clip:
            self.seed.W.array = xp.clip(self.seed.W.array,a_min=-1.0, a_max=1.0)

        # adjust consistency loss update frequency
        self.recon_freq = max(1,int(round(self.args.max_reconst_freq * (step-self.args.reconst_freq_decay_start) / (self.args.iter+1-self.args.reconst_freq_decay_start))))

        ## for discriminator
        fake = None
        if self.args.dis_freq > 0 and ( (step+1) % self.args.dis_freq == 0) and (self.args.lambda_gan+self.args.lambda_adv+self.args.lambda_advs>0):
            # get mini-batch
            if plan is None:
                plan = self.converter(self.get_iterator('planct').next(), self.device)
                plan = losses.add_noise(Variable(plan),self.args.noise_dis)
            
            # create fake
            if self.args.lambda_gan>0:
                if self.args.decoder_only:
                    fake_seed = xp.random.uniform(-1,1,(1,self.args.latent_dim)).astype(np.float32)
                else:
                    fake_seed = self.encoder(xp.random.uniform(-1,1,(1,1,self.args.crop_height,self.args.crop_width)).astype(np.float32))
                fake = self.decoder(fake_seed)
                # decoder
                self.decoder.cleargrads()
                loss_gan = F.average( (self.dis(fake)-1.0)**2 )
                chainer.report({'loss_gan': loss_gan}, self.seed)
                loss_gan *= self.args.lambda_gan
                loss_gan.backward()
                optimizer_dec.update(loss=loss_gan)
                fake_copy = self._buffer.query(fake.array)
            if self.args.lambda_nn>0:
                fake_copy = self._buffer.query(self.converter(self.get_iterator('mvct').next(), self.device))
            if (step+1) % (self.args.iter // 30):
                fake_copy = Variable(self._buffer.query(gen.array))
            # discriminator
            L_real = F.average( (self.dis(plan)-1.0)**2 )
            L_fake = F.average( self.dis(fake_copy)**2 )
            loss_dis = 0.5*(L_real+L_fake)
            self.dis.cleargrads()
            loss_dis.backward()
            optimizer_dis.update()
            chainer.report({'loss_dis': (L_real+L_fake)/2}, self.seed)


        if ((self.iteration+1) % self.args.vis_freq == 0) or  ((step+1)==self.args.iter):
            for i in range(self.args.batchsize):
                outlist=[]
                if not self.args.no_train_seed and not self.args.decoder_only:
                    outlist.append((self.seed()[i],"0sd"))
                if plan_ae is not None:
                    outlist.append((plan[i],'2pl'))
                    outlist.append((plan_ae[i],'3ae'))
                if self.args.lambda_nn>0 or self.args.lambda_adv>0:
                    if self.args.decoder_only:
                        gen_img = self.decoder([self.seed()])
                    else:
                        gen_img = self.decoder(self.encoder(self.seed()))
                    outlist.append((gen_img[i],'1gn'))
                if fake is not None:
                    outlist.append((fake[i],'4fa'))
                for out,typ in outlist:
                    out.to_cpu()
                    HU = (((out+1)/2 * self.args.HU_range)+self.args.HU_base).array  # [-1000=air,0=water,>1000=bone]
                    print("type: ",typ,"HU:",np.min(HU),np.mean(HU),np.max(HU))
                    #visimg = np.clip((out.array+1)/2,0,1) * 255.0
                    b,r = -self.args.HU_range_vis//2,self.args.HU_range_vis
                    visimg = (np.clip(HU,b,b+r)-b)/r * 255.0
                    fn = 'n{:0>5}_iter{:0>6}_p{}_z{}_{}'.format(self.n_reconst,step+1,self.patient_id[i],self.slice[i],typ)
                    write_image(np.uint8(visimg),os.path.join(self.args.out,fn+'.jpg'))
                    if (step+1)==self.args.iter or self.args.save_dcm:
                        #np.save(os.path.join(self.args.out,fn+'.npy'),HU[0])
                        write_dicom(os.path.join(self.args.out,fn+'.dcm'),HU[0])
