"""
Trains CGAN on MaxwellFDFD using Keras
CGAN is Conditional Generative Adversarial Nets.
This version of CGAN is similar to DCGAN. The difference mainly
is that the z-vector of generator is conditioned by a one-hot label
to produce specific fake images.
The discriminator is trained to discriminate real from fake images
that are conditioned on specific one-hot labels.
[1] Radford, Alec, Luke Metz, and Soumith Chintala.
"Unsupervised representation learning with deep convolutional
generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
[2] Mirza, Mehdi, and Simon Osindero. "Conditional generative
adversarial nets." arXiv preprint arXiv:1411.1784 (2014).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers.merge import concatenate
from keras.optimizers import RMSprop
from keras.models import Model
from keras.models import model_from_json
from keras.utils import to_categorical
from keras.models import load_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from PIL import Image
import pandas as pd
import seaborn as sns

# code added for cudnn_status_alloc_failed error
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def load_data():
    # DATAPATH = 'E:/2.CEM/maxwellfdfd_ai/data/train'
    DATAPATH = 'data/train'
    DATASETS = [
        'binary_501',
        'binary_502',
        'binary_503',
        'binary_504',
        'binary_505',
        'binary_506',
        'binary_507',
        'binary_508',
        'binary_509',
        'binary_510',
        'binary_511',
        'binary_512',
        'binary_1001',
        'binary_1002',
        'binary_1003',
        'binary_rl_fix_501',
        'binary_rl_fix_502',
        'binary_rl_fix_503',
        'binary_rl_fix_504',
        'binary_rl_fix_505',
        'binary_rl_fix_506',
        'binary_rl_fix_507',
        'binary_rl_fix_508',
        'binary_rl_fix_509',
        'binary_rl_fix_510',
        'binary_rl_fix_511',
        'binary_rl_fix_512',
        'binary_rl_fix_513',
        'binary_rl_fix_514',
        'binary_rl_fix_515',
        'binary_rl_fix_516',
        'binary_rl_fix_517',
        'binary_rl_fix_518',
        'binary_rl_fix_519',
        'binary_rl_fix_520',
        'binary_rl_fix_1001',
        'binary_rl_fix_1002',
        'binary_rl_fix_1003',
        'binary_rl_fix_1004',
        'binary_rl_fix_1005',
        'binary_rl_fix_1006',
        'binary_rl_fix_1007',
        'binary_rl_fix_1008',
    ]

    x_train = []
    y_train = []

    print('data loading ... ')
    # load dataset
    for data in DATASETS:
        dataframe = pd.read_csv(os.path.join(DATAPATH, data + '.csv'), delim_whitespace=False, header=None)
        dataset = dataframe.values

        # split into input (X) and output (Y) variables
        fileNames = dataset[:, 0]
        y_train.extend(dataset[:, 1:25])
        for idx, file in enumerate(fileNames):
            try:
                image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(int(file))))
            except (TypeError, FileNotFoundError) as te:
                image = Image.open(os.path.join(DATAPATH, data, '{}.tiff'.format(idx + 1)))

            image = image.resize((40, 20))
            image = np.array(image, dtype=np.uint8)
            image = np.where(image > 0, 1, 0)
            x_train.append(image)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    # calculate transmittance
    y_train /= 2767.
    img_shape = (20, 40, 1)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], img_shape[2], img_shape[0], img_shape[1])
    else:
        x_train = x_train.reshape(x_train.shape[0], img_shape[0], img_shape[1], img_shape[2])

    print(x_train.shape[0], 'train samples')
    y_train_df = pd.DataFrame(y_train)
    y_train_df = y_train_df.apply(lambda x: np.argmax(x), axis=1)

    # Countplot
    # y_train_df_for_countplot = pd.DataFrame(y_train_df)
    # y_train_df_for_countplot.columns = ['wavelength (nm)']
    # sns.countplot(x="wavelength (nm)", data=y_train_df_for_countplot)
    # plt.title('Countplot for max transmittance wavelength')
    # plt.show()
    # exit()
    return x_train, y_train_df.values


def build_generator(inputs, labels, image_size):
    """Build a Generator Model
    Inputs are concatenated before Dense layer.
    Stack of BN-ReLU-Conv2DTranpose to generate fake images.
    Output activation is sigmoid instead of tanh in orig DCGAN.
    Sigmoid converges easily.
    # Arguments
        inputs (Layer): Input layer of the generator (the z-vector)
        labels (Layer): Input layer for one-hot vector to condition
            the inputs
        image_size: Target size of one side (assuming square image)
    # Returns
        Model: Generator Model
    """
    image_resize = image_size // 4
    # network parameters
    kernel_size = 5
    layer_filters = [128, 64, 32, 1]

    x = concatenate([inputs, labels], axis=1)
    x = Dense(image_resize * image_resize * 2 * layer_filters[0])(x)
    x = Reshape((image_resize, image_resize * 2, layer_filters[0]))(x)

    for filters in layer_filters:
        # first two convolution layers use strides = 2
        # the last two use strides = 1
        if filters > layer_filters[-2]:
            strides = 2
        else:
            strides = 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same')(x)

    x = Activation('sigmoid')(x)
    # input is conditioned by labels
    generator = Model([inputs, labels], x, name='generator')
    return generator


def build_discriminator(inputs, labels, image_size):
    """Build a Discriminator Model
    Inputs are concatenated after Dense layer.
    Stack of LeakyReLU-Conv2D to discriminate real from fake.
    The network does not converge with BN so it is not used here
    unlike in DCGAN paper.
    # Arguments
        inputs (Layer): Input layer of the discriminator (the image)
        labels (Layer): Input layer for one-hot vector to condition
            the inputs
        image_size: Target size of one side (assuming square image)
    # Returns
        Model: Discriminator Model
    """
    kernel_size = 5
    layer_filters = [32, 64, 128, 256]

    x = inputs

    y = Dense(image_size * image_size * 2)(labels)
    y = Reshape((image_size, image_size * 2, 1))(y)
    x = concatenate([x, y])

    for filters in layer_filters:
        # first 3 convolution layers use strides = 2
        # last one uses strides = 1
        if filters == layer_filters[-1]:
            strides = 1
        else:
            strides = 2
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same')(x)

    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    # input is conditioned by labels
    discriminator = Model([inputs, labels], x, name='discriminator')
    return discriminator


def train(models, data, params):
    """Train the Discriminator and Adversarial Networks
    Alternately train Discriminator and Adversarial networks by batch.
    Discriminator is trained first with properly labelled real and fake images.
    Adversarial is trained next with fake images pretending to be real.
    Discriminator inputs are conditioned by train labels for real images,
    and random labels for fake images.
    Adversarial inputs are conditioned by random labels.
    Generate sample images per save_interval.
    # Arguments
        models (list): Generator, Discriminator, Adversarial models
        data (list): x_train, y_train data
        params (list): Network parameters
    """
    # the GAN models
    generator, discriminator, adversarial = models
    # images and labels
    x_train, y_train = data
    # network parameters
    batch_size, latent_size, train_steps, num_labels, model_name = params
    # the generator image is saved every 500 steps
    save_interval = 500
    # noise vector to see how the generator output evolves during training
    noise_input = np.random.uniform(-1.0, 1.0, size=[25, latent_size])
    # one-hot label the noise will be conditioned to
    noise_class = np.eye(num_labels)[np.arange(0, 25) % num_labels]
    # number of elements in train dataset
    train_size = x_train.shape[0]

    print(model_name,
          "Labels for generated images: ",
          np.argmax(noise_class, axis=1))

    for i in range(train_steps):
        # train the discriminator for 1 batch
        # 1 batch of real (label=1.0) and fake images (label=0.0)
        # randomly pick real images from dataset
        rand_indexes = np.random.randint(0, train_size, size=batch_size)
        real_images = x_train[rand_indexes]

        # print('real_images', real_images.shape)
        # corresponding one-hot labels of real images
        real_labels = y_train[rand_indexes]
        # generate fake images from noise using generator
        # generate noise using uniform distribution
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        # assign random one-hot labels
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]

        # generate fake images conditioned on fake labels
        fake_images = generator.predict([noise, fake_labels])

        # print('fake_images', noise.shape, fake_labels.shape, fake_images.shape)
        # real + fake images = 1 batch of train data
        x = np.concatenate((real_images, fake_images))
        # real + fake one-hot labels = 1 batch of train one-hot labels
        labels = np.concatenate((real_labels, fake_labels))

        # label real and fake images
        # real images label is 1.0
        y = np.ones([2 * batch_size, 1])
        # fake images label is 0.0
        y[batch_size:, :] = 0.0
        # train discriminator network, log the loss and accuracy
        loss, acc = discriminator.train_on_batch([x, labels], y)
        log = "%d: [discriminator loss: %f, acc: %f]" % (i, loss, acc)

        # train the adversarial network for 1 batch
        # 1 batch of fake images conditioned on fake 1-hot labels w/ label=1.0
        # since the discriminator weights are frozen in adversarial network
        # only the generator is trained
        # generate noise using uniform distribution
        noise = np.random.uniform(-1.0, 1.0, size=[batch_size, latent_size])
        # assign random one-hot labels
        fake_labels = np.eye(num_labels)[np.random.choice(num_labels,
                                                          batch_size)]
        # label fake images as real or 1.0
        y = np.ones([batch_size, 1])
        # train the adversarial network
        # note that unlike in discriminator training,
        # we do not save the fake images in a variable
        # the fake images go to the discriminator input of the adversarial
        # for classification
        # log the loss and accuracy
        loss, acc = adversarial.train_on_batch([noise, fake_labels], y)
        log = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
        print(log)
        if (i + 1) % save_interval == 0:
            if (i + 1) == train_steps:
                show = True
            else:
                show = False

            # plot generator images on a periodic basis
            plot_images(generator,
                        noise_input=noise_input,
                        noise_class=noise_class,
                        show=show,
                        step=(i + 1),
                        model_name=model_name)

    # save the model after training the generator
    # the trained generator can be reloaded for future maxwellfdfd image generation
    generator.save('models/{}.h5'.format(model_name))


def get_prediction_model():
    MODEL_JSON_PATH = 'models/cnn_small_rmse_128_300/rmse_rect_1.json'
    MODEL_H5_PATH = 'models/cnn_small_rmse_128_300/rmse_rect_1.h5'

    # load json and create model
    json_file = open(MODEL_JSON_PATH, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(MODEL_H5_PATH)
    return loaded_model


def plot_images(generator,
                noise_input,
                noise_class,
                show=False,
                step=0,
                model_name="gan"):
    """Generate fake images and plot them
    For visualization purposes, generate fake images
    then plot them in a square grid
    # Arguments
        generator (Model): The Generator Model for fake images generation
        noise_input (ndarray): Array of z-vectors
        show (bool): Whether to show plot or not
        step (int): Appended to filename of the save images
        model_name (string): Model name
    """
    r, c = 4, 6
    fig, axs = plt.subplots(r, c, constrained_layout=True)
    cnt = 0

    os.makedirs(model_name, exist_ok=True)
    filename = os.path.join(model_name, "%05d.png" % step)
    images = generator.predict([noise_input, noise_class])
    noise_class_df = pd.DataFrame(noise_class).apply(lambda x: np.argmax(x), axis=1)
    prediction_model = get_prediction_model()
    correct_cnt_top3 = 0
    correct_cnt_top5 = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(images[cnt, :, :, 0], cmap='gray')
            img = images[cnt, :, :, 0].reshape((1,20,40,1))
            real = prediction_model.predict(img)
            argsort_top5 = (-real).argsort()[:, :5][0]
            argsort_top3 = (-real).argsort()[:, :3][0]

            title = '{}nm'.format((noise_class_df[cnt] * 50 + 400))
            # axs[i, j].set_title('{}-{}'.format((noise_class_df[cnt]), (",".join(map(str, argsort_top3)))))

            if noise_class_df[cnt] in argsort_top5:
                title = '{}nm (5)'.format((noise_class_df[cnt] * 50 + 400))
                correct_cnt_top5 += 1

            if noise_class_df[cnt] in argsort_top3:
                title = '{}nm (3)'.format((noise_class_df[cnt] * 50 + 400))
                correct_cnt_top3 += 1

            axs[i, j].set_title(title)
            axs[i, j].axis('off')
            cnt += 1

    accuracy_top3 = round(correct_cnt_top3 / cnt, 2)
    accuracy_top5 = round(correct_cnt_top5 / cnt, 2)
    fig.suptitle("CGAN: Generated images from conditional wavelength \nAccuracy top3: {}, top5: {}\n (3), (5) refers top3, top5".format(accuracy_top3, accuracy_top5), fontsize=11)
    plt.savefig(filename)
    plt.close()

    if show:
        plt.show()
    else:
        plt.close('all')


def build_and_train_models():
    (x_train, y_train) = load_data()

    # reshape data for CNN as (20, 40, 1) and normalize
    image_size = x_train.shape[1]

    num_labels = np.amax(y_train) + 1
    y_train = to_categorical(y_train)

    model_name = "cgan_maxwellfdfd"
    # network parameters
    # the latent or z vector is 100-dim
    latent_size = 100
    batch_size = 64
    train_steps = 40000  # 40000
    lr = 2e-4
    decay = 6e-8
    input_shape = (image_size, image_size * 2, 1)
    label_shape = (num_labels,)

    # build discriminator model
    inputs = Input(shape=input_shape, name='discriminator_input')
    labels = Input(shape=label_shape, name='class_labels')

    discriminator = build_discriminator(inputs, labels, image_size)
    # [1] or original paper uses Adam,
    # but discriminator converges easily with RMSprop
    optimizer = RMSprop(lr=lr, decay=decay)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer,
                          metrics=['accuracy'])
    # discriminator.summary()

    # build generator model
    input_shape = (latent_size,)
    inputs = Input(shape=input_shape, name='z_input')
    generator = build_generator(inputs, labels, image_size)
    # generator.summary()

    # build adversarial model = generator + discriminator
    optimizer = RMSprop(lr=lr * 0.5, decay=decay * 0.5)
    # freeze the weights of discriminator during adversarial training
    discriminator.trainable = False
    outputs = discriminator([generator([inputs, labels]), labels])
    adversarial = Model([inputs, labels],
                        outputs,
                        name=model_name)
    adversarial.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
    # adversarial.summary()

    # train discriminator and adversarial networks
    models = (generator, discriminator, adversarial)
    data = (x_train, y_train)
    params = (batch_size, latent_size, train_steps, num_labels, model_name)
    train(models, data, params)


def test_generator(generator, class_label=None):
    noise_input = np.random.uniform(-1.0, 1.0, size=[25, 100])
    step = 0
    if class_label is None:
        num_labels = 24
        noise_class = np.eye(num_labels)[np.random.choice(num_labels, 25)]
    else:
        noise_class = np.zeros((25, 24))
        noise_class[:, class_label] = 1
        step = class_label

    plot_images(generator,
                noise_input=noise_input,
                noise_class=noise_class,
                show=True,
                step=step,
                model_name="results")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load generator h5 model with trained weights"
    parser.add_argument("-g", "--generator", help=help_)
    help_ = "Specify a specific wavelength to generate 0 ~ 23"
    parser.add_argument("-w", "--wavelength", type=int, help=help_)
    args = parser.parse_args()
    if args.generator:
        generator = load_model(args.generator)
        class_label = None
        if args.wavelength is not None:
            class_label = args.wavelength
        test_generator(generator, class_label)
    else:
        build_and_train_models()