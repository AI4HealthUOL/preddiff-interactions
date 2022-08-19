import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
import sys
import os
import shutil

from .vqvae import vq_encoder_spec, vq_decoder_spec
from .structure_generator import structure_condition_spec, structure_pixelcnn_spec
from .texture_generator import texture_generator_spec, texture_discriminator_spec
from . import nn
from ..imputer_base import ImputerBase
from ...datasets import utils as utils_dataset

class vqvae_imputer(ImputerBase):
    """"
    Based on https://github.com/USTC-JialunPeng/Diverse-Structure-Inpainting
    """
    def __init__(self, full_model_dir,
                    batch_size=1,
                    commitment_cost = 0.25,
                    decay = 0.99,
                    embedding_dim = 64,
                    image_size = 256,
                    nr_attention_s = 4,
                    nr_channel_cond_s = 32,
                    nr_channel_dis_t = 64,
                    nr_channel_gen_t = 64,
                    nr_channel_s = 128,
                    nr_channel_vq = 128,
                    nr_head_s = 8,
                    nr_res_block_vq = 2,
                    nr_res_channel_cond_s = 32,
                    nr_res_channel_s = 128,
                    nr_res_channel_vq = 64,
                    nr_resnet_out_s = 20,
                    nr_resnet_s = 20,
                    num_embeddings = 512,
                    resnet_nonlinearity = "concat_elu"):
        super().__init__()  # no training, use with pre-trained model
        self.imputer_name = 'vq-vae'

        tf.disable_v2_behavior()
        tf.reset_default_graph()

        self.image_size = image_size
        self.embedding_dim = embedding_dim

        self.batch_size = batch_size

        ################### Build structure generator & texture generator ###################
        # Create VQVAE network
        vq_encoder = tf.make_template('vq_encoder', vq_encoder_spec)
        vq_encoder_opt = {'nr_channel': nr_channel_vq, 
                        'nr_res_block':  nr_res_block_vq,
                        'nr_res_channel':  nr_res_channel_vq,
                        'embedding_dim':  embedding_dim,
                        'num_embeddings':  num_embeddings,
                        'commitment_cost':  commitment_cost,
                        'decay':  decay}

        vq_decoder = tf.make_template('vq_decoder', vq_decoder_spec)
        vq_decoder_opt = {'nr_channel':  nr_channel_vq, 
                        'nr_res_block':  nr_res_block_vq,
                        'nr_res_channel':  nr_res_channel_vq,
                        'embedding_dim':  embedding_dim}

        # Create structure generator
        structure_condition = tf.make_template('structure_condition', structure_condition_spec)
        structure_condition_opt = {'nr_channel':  nr_channel_cond_s, 
                                'nr_res_channel':  nr_res_channel_cond_s, 
                                'resnet_nonlinearity':  resnet_nonlinearity}

        structure_pixelcnn = tf.make_template('structure_pixelcnn', structure_pixelcnn_spec)
        structure_pixelcnn_opt = {'nr_channel':  nr_channel_s,
                                'nr_res_channel':  nr_res_channel_s,
                                'nr_resnet':  nr_resnet_s,
                                'nr_out_resnet':  nr_resnet_out_s,
                                'nr_attention':  nr_attention_s,
                                'nr_head':  nr_head_s,
                                'resnet_nonlinearity':  resnet_nonlinearity,
                                'num_embeddings':  num_embeddings}

        # Create texture generator
        texture_generator = tf.make_template('texture_generator', texture_generator_spec)
        texture_generator_opt = {'nr_channel':  nr_channel_gen_t}

        texture_discriminator = tf.make_template('texture_discriminator', texture_discriminator_spec)
        texture_discriminator_opt = {'nr_channel':  nr_channel_dis_t}

        # Sample structure feature maps
        self.top_shape = ( image_size//8,  image_size//8, 1)
        self.img_ph = tf.placeholder(tf.float32, shape=(batch_size,  image_size,  image_size, 3))
        self.mask_ph = tf.placeholder(tf.float32, shape=(batch_size,  image_size,  image_size, 1))
        self.e_sample = tf.placeholder(tf.float32, shape=(batch_size,  image_size//8,  image_size//8,  embedding_dim))
        self.h_sample = tf.placeholder(tf.float32, shape=(batch_size,  image_size//8,  image_size//8, 8* nr_channel_cond_s))

        batch_pos = self.img_ph
        mask = self.mask_ph
        masked = batch_pos * (1. - mask)
        enc_gt = vq_encoder(batch_pos, is_training=False, **vq_encoder_opt)
        dec_gt = vq_decoder(enc_gt['quant_t'], enc_gt['quant_b'], **vq_decoder_opt)
        self.cond_masked = structure_condition(masked, mask, **structure_condition_opt)
        pix_out = structure_pixelcnn(self.e_sample, self.h_sample, dropout_p=0., **structure_pixelcnn_opt)
        pix_out = tf.reshape(pix_out, (-1,  num_embeddings))
        probs_out = tf.nn.log_softmax(pix_out, axis=-1)
        samples_out = tf.multinomial(probs_out, 1)
        samples_out = tf.reshape(samples_out, (-1, ) + self.top_shape[:-1])
        self.new_e_gen = tf.nn.embedding_lookup(tf.transpose(enc_gt['embed_t'], [1, 0]), samples_out, validate_indices=False)

        # Inpaint with generated structure feature maps
        gen_out = texture_generator(masked, mask, self.e_sample, **texture_generator_opt)
        self.img_gen = gen_out * mask + masked * (1. - mask)

        # Discriminator
        dis_out = texture_discriminator(tf.concat([self.img_gen, mask], axis=3), **texture_discriminator_opt)


        # Start session and load model from checkpoint
        # Create a saver to restore full model
        restore_saver = tf.train.Saver()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        # Restore full model
        try:
            full_model_dir = '../checkpoints/imagenet_random'
            ckpt = tf.train.get_checkpoint_state(full_model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                restore_saver.restore(self.sess, ckpt.model_checkpoint_path)
                print('Full model restored ...')
            else:
                print('Restore full model failed! EXIT!')
                raise ImportError
                sys.exit()
        except:
            # root_data_dir = image_data_dir + "CUB_200_2011/"

            # url = 'https://tubcloud.tu-berlin.de/s/i7nMWipCPqKcAF4/download'
            # utils_dataset.download_url(url=url, folder_name='CUB_200_2011')     # file name 'download'
            # file_download = f'{image_data_dir}download'
            # # uncompress .tar.gz file, 'download'
            # with tarfile.open(file_download) as file:
            #     file.extractall(image_data_dir)
            # os.remove(file_download)
            print('expection raised')
            url = 'https://tubcloud.tu-berlin.de/s/nerWLYiNFZPzzbf/download'
            utils_dataset.download_url(url=url, folder_name='vqvae-imputer')  # file name 'download'
            old_file_name = os.getcwd() + '/pred_diff/datasets/Data/vqvae-imputer/download'
            new_dir = '../checkpoints/'
            new_file_name = new_dir + 'imagenet-random.zip'
            os.makedirs(new_dir, exist_ok=True)
            # os.symlink(old_path_image, new_path_image)
            shutil.move(old_file_name, new_file_name)
            os.rmdir(os.getcwd() + '/pred_diff/datasets/Data/vqvae-imputer/')
            from zipfile import ZipFile
            with ZipFile(new_file_name, 'r') as zip:
                print('unzip file')
                zip.extractall(f'{new_dir}/')
            full_model_dir = '../checkpoints/imagenet_random'
            ckpt = tf.train.get_checkpoint_state(full_model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                restore_saver.restore(self.sess, ckpt.model_checkpoint_path)
                print('Full model restored ...')
            else:
                print('Restore full model failed! EXIT!')
                sys.exit()


    def sample_from_model(self, img_np, mask_np):
        cond_masked_np = self.sess.run(self.cond_masked, {self.img_ph: img_np, self.mask_ph: mask_np})
        feed_dict = {self.h_sample: cond_masked_np}
        e_gen = np.zeros((self.batch_size, self.image_size//8, self.image_size//8, self.embedding_dim), dtype=np.float32)
        for yi in range(self.top_shape[0]):
            for xi in range(self.top_shape[1]):
                feed_dict.update({self.e_sample: e_gen})
                new_e_gen_np = self.sess.run(self.new_e_gen, feed_dict)
                e_gen[:,yi,xi,:] = new_e_gen_np[:,yi,xi,:]
        img_gen_np = self.sess.run(self.img_gen, {self.img_ph: img_np, self.mask_ph: mask_np, self.e_sample: e_gen})
        return ((img_gen_np + 1.) * 127.5).astype(np.uint8)
        
    def _impute(self, test_data: np.ndarray, mask_impute: np.ndarray, n_imputations: int):
        """
        test_data: (1, 3, n_pixel, n_pixel), with 3 rgb channels; 0-255 integers
        """
        assert n_imputations%self.batch_size==0, "n_imputations must be a multiple of the batch size"

        n_iter = n_imputations // self.batch_size
        assert test_data.shape[0]==1, 'only available for one sample at a time'
        
        mask_shape = mask_impute.shape

        # bring arrays into tensorflow format HxWx3
        test_data = np.repeat(test_data, self.batch_size, axis=0).transpose((0,2,3,1))
        mask_impute = np.expand_dims(mask_impute[:1,:,:], 0).transpose((0,2,3,1))
        mask_impute = np.repeat(mask_impute, self.batch_size, axis=0)
        
        # Normalize and reshape the image and mask
        test_data = test_data / 127.5 - 1.
        mask_impute = mask_impute*1 #/ 255.

        imputations = np.zeros((n_imputations, 1, test_data.shape[3], test_data.shape[1], test_data.shape[2]))
        for i in range(n_imputations)[::self.batch_size]:
            imputations[i:i+self.batch_size, 0] = self.sample_from_model(test_data, mask_impute).transpose(0,3,1,2)

        return imputations, None
