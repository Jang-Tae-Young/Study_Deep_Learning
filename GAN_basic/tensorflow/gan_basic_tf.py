from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers, datasets
import time
import gan_sub_functions as gsf
from IPython import display


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # 이미지를 [-1, 1]로 정규화합니다.

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# 데이터 배치를 만들고 섞습니다.
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

noise = tf.random.normal([1, 100])

generator = gsf.make_generator_model()
generated_image = generator(noise, training=False)

discriminator = gsf.make_discriminator_model()
decision = discriminator(generated_image)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
print (decision)

generator_optimizer     = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)




EPOCHS = 50
gsf.train(train_dataset, generator, discriminator, EPOCHS, BATCH_SIZE, generator_optimizer, discriminator_optimizer)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


gsf.train(train_dataset, generator, discriminator, EPOCHS, BATCH_SIZE, generator_optimizer, discriminator_optimizer)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

gsf.display_image(EPOCHS)

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob('image*.png')
    filenames = sorted(filenames)
    last = -1
    for i,filename in enumerate(filenames):
        frame = 2*(i**0.5)
        if round(frame) > round(last):
            last = frame
        else:
            continue
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

# import IPython
if IPython.version_info > (6,2,0,''):
    display.Image(filename=anim_file)





