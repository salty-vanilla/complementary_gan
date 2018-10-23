import tensorflow as tf
tf.enable_eager_execution()
import os
import csv
import time
import numpy as np
from PIL import Image
import matplotlib
import scipy.ndimage
matplotlib.use('agg')
import matplotlib.pyplot as plt
from image_sampler import ImageSampler
from ops.losses import pull_away, discriminator_loss
from models import Discriminator, Generator


class Solver(object):
    def __init__(self, generator,
                 discriminator,
                 lr_g: float =1e-3,
                 lr_d: float =1e-3,
                 density_threshold: float =0.05,
                 logdir: str =None):
        self.generator = generator
        self.discriminator = discriminator
        self.opt_g = tf.train.AdamOptimizer(lr_g)
        self.opt_d = tf.train.AdamOptimizer(lr_d)
        self.latent_dim = self.generator.latent_dim
        self.thr = density_threshold
        self.eps = 1e-8

    def _update_discriminator(self, x, z):
        with tf.GradientTape() as tape:
            gz = self.generator(z)
            d_real = self.discriminator(x)
            d_fake = self.discriminator(gz)

            loss_d = discriminator_loss(d_real, d_fake, 'JSD')
            loss_d -= tf.reduce_mean(tf.nn.sigmoid(d_real) * tf.log(tf.nn.sigmoid(d_real)))
        grads = tape.gradient(loss_d, self.discriminator.variables)
        self.opt_d.apply_gradients(zip(grads, self.discriminator.variables))
        return loss_d

    def _update_generator(self, x, z):
        with tf.GradientTape() as tape:
            gz = self.generator(z)
            d_fake, feature_fake = self.discriminator(gz, with_feature=True)
            _, feature_true = self.discriminator(x, with_feature=True)

            loss_fm = tf.reduce_mean((feature_fake - feature_true) ** 2)
            loss_kl = tf.reduce_mean(tf.boolean_mask(tf.log(d_fake),
                                                     d_fake > self.thr))
            loss_pt = pull_away(feature_fake)
            loss_kl -= loss_pt
            loss_g = loss_kl + loss_fm
        grads = tape.gradient(loss_g, self.generator.variables)
        self.opt_g.apply_gradients(zip(grads, self.generator.variables))
        return loss_fm, loss_kl

    def fit(self, x,
            noise_sampler,
            batch_size=64,
            nb_epoch=100,
            visualize_steps=1,
            save_steps=1):
        image_sampler = ImageSampler(normalize_mode='tanh').flow(x,
                                                                 y=None,
                                                                 batch_size=batch_size)
        self.fit_generator(image_sampler,
                           noise_sampler,
                           nb_epoch=nb_epoch,
                           visualize_steps=visualize_steps,
                           save_steps=save_steps)

    def fit_generator(self, image_sampler,
                      noise_sampler,
                      nb_epoch=100,
                      visualize_steps=1,
                      save_steps=1):
        batch_size = image_sampler.batch_size
        nb_sample = image_sampler.nb_sample

        # calc steps_per_epoch
        steps_per_epoch = nb_sample // batch_size
        if nb_sample % batch_size != 0:
            steps_per_epoch += 1

        for epoch in range(1, nb_epoch + 1):
            print('\nepoch {} / {}'.format(epoch, nb_epoch))
            start = time.time()
            for iter_ in range(1, steps_per_epoch + 1):
                x = image_sampler()
                if x.shape[0] != batch_size:
                    continue
                z = noise_sampler(batch_size, self.latent_dim)

                # Discriminator
                x = tf.constant(x, dtype=tf.float32)
                z = tf.constant(z, dtype=tf.float32)
                loss_d = self._update_discriminator(x, z)

                # Generator
                z = noise_sampler(batch_size, self.latent_dim)
                z = tf.constant(z, dtype=tf.float32)
                loss_fm, loss_kl = self._update_generator(x, z)

                print('iter : {} / {}  {:.1f}[s]  loss_d : {:.4f}  loss_fm : {:.4f}  loss_kl : {:.4f} \r'
                      .format(iter_, steps_per_epoch, time.time() - start,
                              loss_d, loss_fm, loss_kl), end='')
