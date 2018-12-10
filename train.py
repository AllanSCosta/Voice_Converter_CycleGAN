import os
import numpy as np
import argparse
import time

from preprocess import *
from model import GAN

should_preprocess = False

train_A_dir_default = './data/obama'
model_dir_default = './model/'
model_name_default = 'model.ckpt'
random_seed_default = 0
output_dir_default = './validation_output'
tensorboard_log_dir_default = './log'
preprocessed_data_dir = "./preprocessed_data/"


def train(train_dir, model_dir, model_name, random_seed, output_dir):
    np.random.seed(random_seed)

    num_epochs = 1000
    mini_batch_size = 1
    generator_learning_rate = 0.0002
    generator_learning_rate_decay = generator_learning_rate / 200000
    discriminator_learning_rate = 0.0001
    discriminator_learning_rate_decay = discriminator_learning_rate / 200000
    sampling_rate = 16000
    num_mcep = 24
    frame_period = 5.0
    n_frames = 128

    if should_preprocess:
        print('Preprocessing Data...')
        train_dir = train_A_dir_default

        start_time = time.time()

        if not os.path.exists(preprocessed_data_dir):
            os.makedirs(preprocessed_data_dir)

        wavs = load_wavs(wav_dir=train_dir, sr=sampling_rate)
        f0s, timeaxes, sps, aps, coded_sps = world_encode_data(wavs=wavs, fs=sampling_rate,
                                                               frame_period=frame_period,
                                                               coded_dim=num_mcep)
        np.save(preprocessed_data_dir + "f0s", np.asarray(f0s))
        np.save(preprocessed_data_dir + "timeaxes", np.asarray(timeaxes))
        np.save(preprocessed_data_dir + "sps", np.asarray(sps))
        np.save(preprocessed_data_dir + "aps", np.asarray(aps))
        np.save(preprocessed_data_dir + "coded_sps", np.asarray(coded_sps))

        log_f0s_mean, log_f0s_std = logf0_statistics(f0s)
        np.save(preprocessed_data_dir + "log_f0s_mean", np.asarray(log_f0s_mean))
        np.save(preprocessed_data_dir + "log_f0s_std", np.asarray(log_f0s_std))


        print('Log Pitch A')
        print('Mean: %f, Std: %f' % (log_f0s_mean, log_f0s_std))

        end_time = time.time()
        time_elapsed = end_time - start_time
        print('Preprocessing Done.')
        print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (
            time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))

    else:

        print("Loading Preprocessed Data")
        f0s = np.load(preprocessed_data_dir + "f0s.npy")
        timeaxes = np.load(preprocessed_data_dir + "timeaxes.npy")
        sps = np.load(preprocessed_data_dir + "sps.npy")
        aps = np.load(preprocessed_data_dir + "aps.npy")
        coded_sps = np.load(preprocessed_data_dir + "coded_sps.npy")
        log_f0s_mean = np.load(preprocessed_data_dir + "log_f0s_mean.npy")
        log_f0s_std = np.load(preprocessed_data_dir + "log_f0s_std.npy")


    coded_sps_transposed = transpose_in_list(lst=coded_sps)
    coded_sps_norm, coded_sps_mean, coded_sps_std = coded_sps_normalization_fit_transoform(
        coded_sps=coded_sps_transposed)
    print("Input data fixed.")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    np.savez(os.path.join(model_dir, 'logf0s_normalization.npz'), mean_A=log_f0s_mean, std_A=log_f0s_std)
    np.savez(os.path.join(model_dir, 'mcep_normalization.npz'), mean_A=coded_sps_mean, std_A=coded_sps_std)


    model = GAN(num_features=num_mcep)
    for epoch in range(num_epochs):
        print('Epoch: %d' % epoch)
        start_time_epoch = time.time()

        dataset = sample_single_train_data(dataset=coded_sps_norm, n_frames=n_frames)
        n_samples = dataset.shape[0]

        for i in range(n_samples // mini_batch_size):
            num_iterations = n_samples // mini_batch_size * epoch + i
            if num_iterations > 200000:
                generator_learning_rate = max(0, generator_learning_rate - generator_learning_rate_decay)
                discriminator_learning_rate = max(0.0, discriminator_learning_rate - discriminator_learning_rate_decay)

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            generator_loss, discriminator_loss, generation = model.train(input=dataset[start:end],
                                                             generator_learning_rate=generator_learning_rate,
                                                             discriminator_learning_rate=discriminator_learning_rate)

            if i % 50 == 0:
                print('Iteration: {:07d}, Generator Learning Rate: {:.7f}, Discriminator Learning Rate: {:.7f}, \
                       Generator Loss : {:.3f}, Discriminator Loss : {:.3f}'
                      .format(num_iterations, generator_learning_rate,
                              discriminator_learning_rate, generator_loss, discriminator_loss))

            if i == (n_samples // mini_batch_size) - 1 and epoch % 20 == 0:
                print("")
                print("---------------------------------------------------------------")
                print('Generating Validation Data from noise...')
                save_generated_voice(generation, output_dir, epoch // 20)

        model.save(directory=model_dir, filename=model_name)

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch
        print('Time Elapsed for This Epoch: %02d:%02d:%02d' % (
            time_elapsed_epoch // 3600, (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))


def save_generated_voice(generation, output_dir, epoch):
    librosa.output.write_wav(os.path.join(output_dir, "name" + str(epoch) + ".wav"), generation[0], 16000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CycleGAN model for datasets.')

    parser.add_argument('--train_A_dir', type=str, help='Directory for A.', default=train_A_dir_default)
    parser.add_argument('--model_dir', type=str, help='Directory for saving models.', default=model_dir_default)
    parser.add_argument('--model_name', type=str, help='File name for saving model.', default=model_name_default)
    parser.add_argument('--random_seed', type=int, help='Random seed for model training.', default=random_seed_default)
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for converted validation voices.', default=output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type=str,
                        help='TensorBoard log directory.', default=tensorboard_log_dir_default)

    argv = parser.parse_args()

    train_dir = argv.train_A_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    output_dir = argv.output_dir
    tensorboard_log_dir = argv.tensorboard_log_dir

    train(train_dir=train_dir, model_dir=model_dir, model_name=model_name,
          random_seed=random_seed,
          output_dir=output_dir)
