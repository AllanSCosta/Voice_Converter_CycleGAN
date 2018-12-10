import os
import tensorflow as tf
from module import discriminator, generator_gatedcnn
from utils import l2_loss
from datetime import datetime
import numpy as np


class GAN(object):

    def __init__(self, num_features, discriminator=discriminator,
                 generator=generator_gatedcnn, mode='train', log_dir='./log'):
        self.num_features = num_features
        # [batch_size, num_features, num_frames]
        self.input_shape = [None, num_features, None]

        self.discriminator = discriminator
        self.generator = generator
        self.mode = mode

        self.build_model()
        self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))
            self.writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
            self.generator_summaries, self.discriminator_summaries = self.summary()

    def build_model(self):
        print("building model")

        # Placeholders noisy data
        self.noise = tf.placeholder(tf.float32, shape=self.input_shape, name='noise')
        # Placeholders for real training samples
        self.input_real = tf.placeholder(tf.float32, shape=self.input_shape, name='input_real')
        # Placeholders for fake generated samples
        self.input_fake = tf.placeholder(tf.float32, shape=self.input_shape, name='input_fake')
        # Placeholder for test samples
        self.input_test = tf.placeholder(tf.float32, shape=self.input_shape, name='input_test')

        self.generation = self.generator(inputs=self.noise, reuse=False, scope_name='generator')
        self.discrimination = self.discriminator(inputs=self.generation, reuse=False, scope_name='discriminator')

        # Generator wants to fool discriminator
        self.generator_loss = l2_loss(y=tf.ones_like(self.discrimination), y_hat=self.discrimination)

        # Merge the two generators and the cycle loss
        self.generator_loss = self.generator_loss

        # Discriminator loss
        self.discrimination_input_real = self.discriminator(inputs=self.input_real,
                                                            reuse=True, scope_name='discriminator')
        self.discrimination_input_fake = self.discriminator(inputs=self.input_fake,
                                                            reuse=True, scope_name='discriminator')

        self.discriminator_loss_input_real = l2_loss(y=tf.ones_like(self.discrimination_input_real),
                                                     y_hat=self.discrimination_input_real)
        self.discriminator_loss_input_fake = l2_loss(y=tf.zeros_like(self.discrimination_input_fake),
                                                     y_hat=self.discrimination_input_fake)

        self.discriminator_loss = (self.discriminator_loss_input_real + self.discriminator_loss_input_fake) / 2

        # Merge the two discriminators into one
        self.discriminator_loss = self.discriminator_loss

        # Categorize variables because we have to optimize the two sets of the variables separately
        trainable_variables = tf.trainable_variables()
        self.discriminator_vars = [var for var in trainable_variables if 'discriminator' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'generator' in var.name]

        self.generation_test = self.generator(inputs=self.input_test, reuse=True, scope_name='generator')

    def optimizer_initializer(self):
        self.generator_learning_rate = tf.placeholder(tf.float32, None, name='generator_learning_rate')
        self.discriminator_learning_rate = tf.placeholder(tf.float32, None, name='discriminator_learning_rate')
        discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=self.discriminator_learning_rate, beta1=0.5)
        self.discriminator_optimizer = discriminator_optimizer.minimize(self.discriminator_loss,
                                                                        var_list=self.discriminator_vars)
        generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.generator_learning_rate, beta1=0.5)
        self.generator_optimizer = generator_optimizer.minimize(self.generator_loss, var_list=self.generator_vars)

    def train(self, input, generator_learning_rate, discriminator_learning_rate):
        generation, generator_loss, _, generator_summaries = self.sess.run(
            [self.generation, self.generator_loss, self.generator_optimizer, self.generator_summaries],
            feed_dict={self.noise: np.random.normal(0, 1, input.shape),
                       self.generator_learning_rate: generator_learning_rate})

        self.writer.add_summary(generator_summaries, self.train_step)

        discriminator_loss, _, discriminator_summaries = self.sess.run(
            [self.discriminator_loss, self.discriminator_optimizer, self.discriminator_summaries],
            feed_dict={self.input_real: input,
                       self.input_fake: generation,
                       self.discriminator_learning_rate: discriminator_learning_rate
                       })

        self.writer.add_summary(discriminator_summaries, self.train_step)

        self.train_step += 1
        return generator_loss, discriminator_loss, generation

    def test(self):
        generation = self.sess.run(self.generation_test)
        return generation

    def save(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))

        return os.path.join(directory, filename)

    def load(self, file_path):
        self.saver.restore(self.sess, file_path)

    def summary(self):
        with tf.name_scope('generator_summaries'):
            generator_loss_summary = tf.summary.scalar('generator_loss', self.generator_loss)
            generator_summaries = tf.summary.merge([generator_loss_summary])

        with tf.name_scope('discriminator_summaries'):
            discriminator_loss_summary = tf.summary.scalar('discriminator_loss', self.discriminator_loss)
            discriminator_summaries = tf.summary.merge([discriminator_loss_summary])

        return generator_summaries, discriminator_summaries

if __name__ == '__main__':
    model = GAN(num_features=24)
    print('Graph Compile Succeeded.')
