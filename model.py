import numpy as np
import tensorflow as tf
import encoder as enc
import preprocessing as pr
from cv2 import imwrite
from tensorflow.python.framework import ops
from random import uniform

class Model:
    def __init__(self, output_dim):
        #self.model_session = tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)))
        self.output_dim = output_dim #Assumes we are attempting to generate a square image
        self.caption_arr = pr.construct_caption_arr(pr.num_to_attr('data/LabelledBirds/attributes/attributes.txt'), 'data/LabelledBirds/attributes/image_attribute_labels.txt')
        self.stage_1_vars = []
        self.text2vec = enc.load_model()
        self.batch_size = 28

    def encode(self, text):
        return self.text2vec.predict(enc.tokenize(text))
    def generator_1(self, text_input, is_train = True):
        with tf.variable_scope('g1', reuse = tf.AUTO_REUSE) as scope:
            # print(text_input.shape)
            noise_vec = tf.random_normal(shape = text_input.shape)
            input = tf.concat([text_input, noise_vec], 1)

            W1 = tf.get_variable('gen1', shape = (input.shape[1].value, 4*4*self.output_dim), dtype = tf.float32, initializer = tf.truncated_normal_initializer)
            B1 = tf.get_variable('gen2', shape = (4*4*self.output_dim), dtype = tf.float32, initializer = tf.constant_initializer(0.0))
            Y1 = tf.add(tf.matmul(input, W1), B1)
            act_ffn = tf.nn.leaky_relu(Y1, alpha = .01)

            reshaped = tf.reshape(act_ffn, shape = [-1, 4, 4, self.output_dim])
            output_dim = self.output_dim / 4

            start = int(self.output_dim/2)

        #print(reshaped)
            conv1 = tf.layers.conv2d_transpose(reshaped, start, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5), name = 'gen3') #Add filters,
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act1 = tf.nn.leaky_relu(bn1, alpha = .01)

            conv2 = tf.layers.conv2d_transpose(conv1, int(start/2), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5), name = 'gen4') #Add filters,
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act2 = tf.nn.leaky_relu(bn2, alpha = .01)

            conv3 = tf.layers.conv2d_transpose(conv2, int(start/4), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5), name = 'gen5') #Add filters,
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None )
            act3 = tf.nn.leaky_relu(bn3, alpha = .01)

            conv4 = tf.layers.conv2d_transpose(conv3, 3, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5), name = 'gen6') #Add filters,
            #bn4 = tf.contrib.layers.batch_norm(conv4, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act4 = tf.nn.tanh(conv4)

            if is_train:
                return act4
                # return tf.squeeze(act4) #should be a tensor of shape (output_dim, output_dim, 3)
            #return tf.squeeze(conv4)
            return conv4

    #Input is your image. Discriminator is only used during training, so you add the input when training
    def discriminator_1(self, input_img, encoded_txt, is_train = True, batch_size = 28):
        #Shrink to lower dimension
        with tf.variable_scope('d1', reuse = tf.AUTO_REUSE) as scope:
            #Reshape/Resize Text Embedding
            output_dim = 1
            N, M = 16, 32 #Play around with these values. Likely, larger the value, more detail is retained but slower the training is

            W0 = tf.get_variable('dis1', shape = (encoded_txt.shape[1], M*M*N), dtype = tf.float32, initializer = tf.truncated_normal_initializer)
            B0 = tf.get_variable('dis2', shape = (M*M*N,), dtype = tf.float32, initializer = tf.constant_initializer(0.0))
            Y0 = tf.nn.leaky_relu(tf.add(tf.matmul(encoded_txt, W0), B0))

            #Expand to create MxMxN 3d tensor
            mod_encoding = tf.reshape(Y0, shape = (batch_size, M, M, N))

            input_conv1 = tf.layers.conv2d(tf.to_float(input_img), int(N), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'dis3')
            input_bn1 = tf.contrib.layers.batch_norm(input_conv1, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act1 = tf.nn.leaky_relu(input_bn1, alpha = .01)

            #Concatenating img and text tensors, then preform convolutions
            input_concat = tf.concat([mod_encoding, input_act1], 2)
            print(input_concat.shape)

            # input_concat = tf.reshape(input_concat, shape = lst)
            conv1 = tf.layers.conv2d(input_concat, 32, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv1dis4')
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            act1 = tf.nn.leaky_relu(bn1, alpha = .01)

            conv2 = tf.layers.conv2d(act1, 64, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv2dis5')
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act2 = tf.nn.leaky_relu(bn2, alpha = .01)

            conv3 = tf.layers.conv2d(act2, 128, kernel_size = [5, 5], strides = (2, 2), padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv3dis6')
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act3 = tf.nn.leaky_relu(bn3, alpha = .01)

            conv4 = tf.layers.conv2d(act3, 256, kernel_size = [5, 5], strides = (2,2), padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv4dis7')
            bn4 = tf.contrib.layers.batch_norm(conv4, decay = 0.9, epsilon = 1e-5, is_training = is_train, updates_collections = None)
            act4 = tf.nn.leaky_relu(bn4, alpha = .01)

                #Feed Forward Network
            dim = int(np.prod(act4.get_shape()[1:]))
            h = tf.reshape(act4, shape = [-1, dim])

            W1 = tf.get_variable(shape = [dim, 128], dtype = tf.float32, name = 'dis7')
            B1 = tf.get_variable(shape = [128], dtype = tf.float32, name = 'dis8')
            Y1 = tf.nn.leaky_relu(tf.add(tf.matmul(h, W1), B1), alpha = .01)

            W2 = tf.get_variable(shape = [128, 1], dtype = tf.float32, name = 'dis9')
            B2 = tf.get_variable(shape = [1], dtype = tf.float32, name = 'dis10')
            Y2 = tf.add(tf.matmul(Y1, W2), B2)
            return tf.nn.sigmoid(Y2), Y2

    def train_1(self, save = True):
        #tf.reset_default_graph()
        batch_size = self.batch_size
        text_input = tf.placeholder(dtype = tf.float32)
        images = tf.placeholder(dtype = tf.float32)
        lr = tf.placeholder(dtype = tf.float32)

        if text_input.shape:
            input_caps = text_input
            input_imgs = images
        else:
            imgs = []
            caps = []
            feeder = pr.FeedExamples()
            for k in range(batch_size):
                # print('one')
                train_data = feeder.next_example()
                train_image = train_data[0]
                txt = train_data[1]
                imgs.append(train_image)
                caps.append(txt)
            input_imgs = None
            input_caps = None
            for i in range(batch_size):
                path = tf.read_file(imgs[i])
                img = tf.cast(tf.image.decode_jpeg(path, channels = 3), dtype = tf.float32)
                resized_img = tf.image.resize_images(img, (64, 64))
                sess = tf.InteractiveSession()
                real_image = resized_img.eval()
                sess.close()
                new_shape = [1, 64, 64, 3]
                real_image = np.reshape(real_image, new_shape)
                # print(real_image.shape)
                if input_imgs is None:
                    input_imgs = np.array(real_image)
                else:
                    input_imgs = np.concatenate((input_imgs, real_image), axis = 0)
                    # print(input_imgs.shape)

                if input_caps is None:
                    input_caps = np.array(self.encode(caps[i]))
                    print(input_caps.shape)
                else:
                    input_caps = np.vstack((input_caps, self.encode(caps[i])))
                    print(input_caps.shape)

        batch = tf.nn.tanh(input_imgs)
        fake_images = self.generator_1(input_caps)
        # print(input_caps.shape)
        # print(input_imgs.shape)

        print('First')
        fake_result, fr_logits = self.discriminator_1(fake_images, input_caps)
        print('Second')
        real_result_fake_caption, rrfc_logits = self.discriminator_1(batch, tf.random_shuffle(input_caps))
        print('Third')
        real_result_real_caption, rrrc_logits = self.discriminator_1(batch, input_caps)

        FR = tf.reduce_mean(fake_result)

        # Added noise to the labels to improve training and establish a proper equilibrium
        dis_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = rrrc_logits, labels = tf.constant(np.array([[0.7 + (uniform(0, 1)*0.5) for _ in range(batch_size)]]).T, dtype = tf.float32)) + tf.nn.sigmoid_cross_entropy_with_logits(logits = rrfc_logits, labels = tf.constant(np.array([[uniform(0, 0.3) for _ in range(batch_size)]]).T, dtype = tf.float32)) +\
        tf.nn.sigmoid_cross_entropy_with_logits(logits = fr_logits, labels = tf.constant(np.array([[uniform(0, 0.3) for _ in range(batch_size)]]).T, dtype = tf.float32))
        gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = fr_logits, labels = tf.constant(np.array([[0.7 + (uniform(0, 1)*0.5) for _ in range(batch_size)]]).T, dtype = tf.float32))

        dis_loss = tf.reduce_mean(dis_loss)
        gen_loss = tf.reduce_mean(gen_loss)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd1' in var.name]
        g_vars = [var for var in t_vars if 'g1' in var.name]

        with tf.variable_scope('gan1', reuse = tf.AUTO_REUSE):
            trainer_gen = tf.train.AdamOptimizer(learning_rate = lr, name = 'stage1gen').minimize(gen_loss, var_list = g_vars)
            trainer_dis = tf.train.AdamOptimizer(learning_rate = lr, name = 'stage1dis').minimize(dis_loss, var_list = d_vars)

        with tf.Session() as sess:
            run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
            num_of_imgs = 11788
            num_epochs = 1000 #adjust if necessary
            sess.run(tf.local_variables_initializer(), options = run_opts)
            sess.run(tf.global_variables_initializer(), options = run_opts)
            # saver = tf.train.Saver()
            # saver.restore(sess, tf.train.latest_checkpoint('ckpts'))

            print('Start Training::: ')
            for i in range(num_epochs):
                print()
                print()
                print(str(i+1) + 'th epoch: ')
                print()
                print()
                feeder = pr.FeedExamples()
                num_of_batches = int(num_of_imgs/batch_size)
                for _ in range(num_of_batches):
                    input_imgs, input_caps = feeder.next_batch(batch_size, self.encode, 64)
                    l_r = 2e-4 * (0.5 ** (i//100))
                    if l_r == 0.0:
                        print('Not training, fix Learning Rate')
                    dLoss, FR_dis, op = sess.run([dis_loss, FR, trainer_dis], feed_dict = {text_input : input_caps, images : input_imgs, lr : l_r})
                    gLoss, op = sess.run([gen_loss, trainer_gen], feed_dict = {text_input : input_caps, images : input_imgs, lr : l_r})
                    print('Generator Loss: ' + str(gLoss))
                    print('Discriminator Loss: ' + str(dLoss))
                    print('Fake Image: ' + str(FR_dis))
                #Save current state after every epoch
                if save:
                    print('Saving:')
                    saver = tf.train.Saver()
                    saver.save(sess, "./ckpts/model.ckpt")

    def generator_2(self, text_encoding, is_train = True):
        #Upsample text embedding
        #Play around with these values. Likely, larger the value, more detail is retained but slower the training is
        with tf.variable_scope('g2', reuse = tf.AUTO_REUSE) as scope:
            N, M = 32, 16

            W0 = tf.get_variable('w0', shape = (text_encoding.shape[1], M*M*N), dtype = tf.float32, initializer = tf.truncated_normal_initializer)
            B0 = tf.get_variable('b0', shape = (M*M*N,), dtype = tf.float32, initializer = tf.constant_initializer(0.0))
            Y0 = tf.nn.leaky_relu(tf.add(tf.matmul(text_encoding, W0), B0), alpha = .01)

            #Expand to create MxMxN 3d tensor
            mod_embedding = tf.reshape(Y0, shape = (M, M, N))

            #Downsample gan1 image
            gan1img = self.generator_1(text_encoding)
            input_img = tf.reshape(gan1img, shape = (1, 64, 64, 3))

            input_conv1 = tf.layers.conv2d(input_img, N, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02))
            input_bn1 = tf.contrib.layers.batch_norm(input_conv1, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act1 = tf.nn.leaky_relu(input_bn1, alpha = .01)

            input_conv2 = tf.layers.conv2d(input_act1, N, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02))
            input_bn2 = tf.contrib.layers.batch_norm(input_conv2, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act2 = tf.nn.leaky_relu(input_bn2, alpha = .01)

            conv_input = tf.concat([mod_embedding, tf.squeeze(input_act2)], 2)

            lst = [1]
            lst.extend(conv_input.shape)
            input = tf.reshape(conv_input, shape = lst)
            #Deconvolution to get high quality image
            conv1 = tf.layers.conv2d_transpose(input, 60, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5)) #Add filters,
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act1 = tf.nn.leaky_relu(bn1, alpha = .01)

            conv2 = tf.layers.conv2d_transpose(act1, 40, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5)) #Add filters,
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act2 = tf.nn.leaky_relu(bn2, alpha = .01)

            conv3 = tf.layers.conv2d_transpose(act2, 20, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5)) #Add filters,
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act3 = tf.nn.leaky_relu(bn3, alpha = .01)

            conv4 = tf.layers.conv2d_transpose(act3, 3, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5)) #Add filters,
            bn4 = tf.contrib.layers.batch_norm(conv4, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act4 = tf.nn.leaky_relu(bn4, alpha = .01)

            return act4

    def discriminator_2(self, input_img, text_encoding, is_train = True):
        with tf.variable_scope('d2', reuse = tf.AUTO_REUSE) as scope:
            output_dim = 1
            N, M = 16, 16 #Play around with these values. Likely, larger the value, more detail is retained but OOM Error may be thrown.
            W0 = tf.get_variable('w0', shape = (init_embedding.shape[1], M*M*N), dtype = tf.float32, initializer = tf.truncated_normal_initializer)
            B0 = tf.get_variable('b0', shape = (M*M*N,), dtype = tf.float32, initializer = tf.constant_initializer(0.0))
            Y0 = tf.nn.leaky_relu(tf.add(tf.matmul(init_embedding, W0), B0), alpha = .01)

            #Expand to create MxMxN 3d tensor
            mod_embedding = tf.reshape(Y0, shape = (M, M, N))

            #Convolution for img
            resized_img = tf.image.resize_images(input_img, (256,256))
            if len(resized_img.shape) < 4:
                lst = [1]
                for i in resized_img.shape:
                    lst.append(i)
                resized_img = tf.reshape(resized_img, shape = lst)

            input_conv1 = tf.layers.conv2d(tf.to_float(resized_img), int(N/4), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02))
            input_bn1 = tf.contrib.layers.batch_norm(input_conv1, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act1 = tf.nn.leaky_relu(input_bn1, alpha = .01)

            input_conv2 = tf.layers.conv2d(input_act1, int(N/2), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02))
            input_bn2 = tf.contrib.layers.batch_norm(input_conv2, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act2 = tf.nn.leaky_relu(input_bn2, alpha = .01)

            input_conv3 = tf.layers.conv2d(input_act2, 3*int(N/4), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02))
            input_bn3 = tf.contrib.layers.batch_norm(input_conv3, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act3 = tf.nn.leaky_relu(input_bn3, alpha = .01)

            input_conv4 = tf.layers.conv2d(input_act3, N, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02))
            input_bn4 = tf.contrib.layers.batch_norm(input_conv4, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act4 = tf.nn.leaky_relu(input_bn4, alpha = .01)
            #Concatenating img and text tensors, then preform convolutions
            input_concat = tf.concat([mod_embedding, tf.squeeze(input_act4)], 2)

            lst = [1]
            lst.extend(input_concat.shape)
            img = tf.reshape(input_concat, shape = lst)
            conv1 = tf.layers.conv2d(img, 128, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv1')
            #Preform batch normalization because it was found to improve GAN preformance
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            act1 = tf.nn.leaky_relu(bn1, alpha = .01)

            conv2 = tf.layers.conv2d(act1, 256, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv2')
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act2 = tf.nn.leaky_relu(bn2, alpha = .01)

            conv3 = tf.layers.conv2d(act2, 512, kernel_size = [5, 5], strides = (2, 2), padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv3')
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act3 = tf.nn.leaky_relu(bn3, alpha = .01)

            #Feed Forward Network
            dim  = int(np.prod(act3.get_shape()[1:]))#1: because of batch size
            h = tf.reshape(act3, shape = [-1, dim])

            W1 = tf.get_variable('dis2w1', shape = [dim, 128], dtype = tf.float32)
            B1 = tf.get_variable('dis2b1', shape = [128], dtype = tf.float32)
            Y1 = tf.nn.leaky_relu(tf.add(tf.matmul(h, W1), B1), alpha = .01)

            W2 = tf.get_variable('dis2w2', shape = [128, 1], dtype = tf.float32)
            B2 = tf.get_variable('dis2b2', shape = [1], dtype = tf.float32)
            Y2 = tf.add(tf.matmul(Y1, W2), B2) #Use sigmoid activation because you want probability output
            return tf.nn.sigmoid(Y2)
        #Shrink to lower dimension



    def train_2(self, save = True):
        batch_size = self.batch_size
        text_input = tf.placeholder(dtype = tf.float32)
        images = tf.placeholder(dtype = tf.float32)
        lr = tf.placeholder(dtype = tf.float32)

        if text_input.shape:
            input_caps = text_input
            input_imgs = images
        else:
            feeder = pr.FeedExamples(shrink_imgs = False)
            input_imgs, input_caps = feeder.next_batch(batch_size = 28, encode = self.encode, img_size = 256)

        batch = tf.nn.tanh(input_imgs)
        fake_images = self.generator_2(input_caps)
        # print(input_caps.shape)
        # print(input_imgs.shape)

        print('First')
        fake_result, fr_logits = self.discriminator_1(fake_images, input_caps)
        print('Second')
        real_result_fake_caption, rrfc_logits = self.discriminator_1(batch, tf.random_shuffle(input_caps))
        print('Third')
        real_result_real_caption, rrrc_logits = self.discriminator_1(batch, input_caps)

        FR = tf.reduce_mean(fake_result)

        # Added noise to the labels to improve training and establish a proper equilibrium
        dis_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = rrrc_logits, labels = tf.constant(np.array([[0.7 + (uniform(0, 1)*0.5) for _ in range(batch_size)]]).T, dtype = tf.float32)) + tf.nn.sigmoid_cross_entropy_with_logits(logits = rrfc_logits, labels = tf.constant(np.array([[uniform(0, 0.3) for _ in range(batch_size)]]).T, dtype = tf.float32)) +\
        tf.nn.sigmoid_cross_entropy_with_logits(logits = fr_logits, labels = tf.constant(np.array([[uniform(0, 0.3) for _ in range(batch_size)]]).T, dtype = tf.float32))
        gen_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = fr_logits, labels = tf.constant(np.array([[0.7 + (uniform(0, 1)*0.5) for _ in range(batch_size)]]).T, dtype = tf.float32))

        dis_loss = tf.reduce_mean(dis_loss)
        gen_loss = tf.reduce_mean(gen_loss)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd2' in var.name]
        g_vars = [var for var in t_vars if 'g2' in var.name]

        with tf.variable_scope('gan2', reuse = tf.AUTO_REUSE):
            trainer_gen = tf.train.AdamOptimizer(learning_rate = lr, name = 'stage1gen').minimize(gen_loss, var_list = g_vars)
            trainer_dis = tf.train.AdamOptimizer(learning_rate = lr, name = 'stage1dis').minimize(dis_loss, var_list = d_vars)

        with tf.Session() as sess:
            run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
            num_of_imgs = 11788
            num_epochs = 1000 #adjust if necessary
            sess.run(tf.local_variables_initializer(), options = run_opts)
            sess.run(tf.global_variables_initializer(), options = run_opts)
            # saver = tf.train.Saver()
            # saver.restore(sess, tf.train.latest_checkpoint('ckpts'))

            print('Start Training::: ')
            for i in range(num_epochs):
                print()
                print()
                print(str(i+1) + 'th epoch: ')
                print()
                print()
                feeder = pr.FeedExamples()
                num_of_batches = int(num_of_imgs/batch_size)
                for _ in range(num_of_batches):
                    input_imgs, input_caps = feeder.next_batch(batch_size, self.encode, 64)
                    l_r = 2e-4 * (0.5 ** (i//100))
                    if l_r == 0.0:
                        print('Not training, fix Learning Rate')
                    dLoss, FR_dis, op = sess.run([dis_loss, FR, trainer_dis], feed_dict = {text_input : input_caps, images : input_imgs, lr : l_r})
                    gLoss, op = sess.run([gen_loss, trainer_gen], feed_dict = {text_input : input_caps, images : input_imgs, lr : l_r})
                    print('Generator Loss: ' + str(gLoss))
                    print('Discriminator Loss: ' + str(dLoss))
                    print('Fake Image: ' + str(FR_dis))
                #Save current state after every epoch
                if save:
                    print('Saving:')
                    saver = tf.train.Saver()
                    saver.save(sess, "./ckpts/model.ckpt")

    def normalize_image(self, init_img):
        new = np.reshape(init_img, np.prod(init_img.shape))
        m = np.sum(new)/np.prod(init_img.shape)
        s = np.std(new)
        out_img = np.zeros(init_img.shape)
        for i in range(len(init_img)):
            for j in range(len(init_img[0])):
                for k in range(len(init_img[0][0])):
                    out_img[i][j][k] = (init_img[i][j][k] - m)/s
        return out_img
    #Need this function because OpenCV decodes in BGR order, not RGB
    def flip_channel_order(self, np_img, img_dim = 256):
        one = np.zeros(shape = (img_dim, img_dim, 1))
        two = np.zeros(shape = (img_dim, img_dim, 1))
        three = np.zeros(shape = (img_dim, img_dim, 1))
        for i in range(img_dim):
            for j in range(img_dim):
                for k in range(3):
                    if k == 0:
                        one[i][j][0] = np_img[i][j][k]
                    if k == 1:
                        two[i][j][0] = np_img[i][j][k]
                    if k == 2:
                        three[i][j][0] = np_img[i][j][k]
        a = tf.constant(one, dtype = tf.uint8)
        b = tf.constant(two, dtype = tf.uint8)
        c = tf.constant(three, dtype = tf.uint8)
        tf_img = tf.concat([three, two, one], axis = 2)
        sess = tf.InteractiveSession()
        np_img1 = tf_img.eval()
        sess.close()
        return np_img1


    def predict(self, input_text):
        tf.reset_default_graph()
        init_img = self.generator_2(input_text)
        d = self.discriminator_2(init_img, input_text, is_train = False).eval()
        tensor_img = tf.squeeze(tf.cast(init_img, dtype = tf.uint8))
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('ckpts'))
            d, np_img = sess.run([d, tensor_img])
            print(d)
            imwrite("output_image.jpg", self.flip_channel_order(np_img))

    def predict_lowres(self, input_text, exists = False):
        tf.reset_default_graph()
        text_encoding = self.encode(input_text)
        init_img = self.generator_1(text_encoding, is_train = False)
        print()
        d = self.discriminator_1(init_img, text_encoding, is_train = False, batch_size = 1)
        tensor_img = tf.squeeze(init_img)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('ckpts'))
            a = sess.run('g1/gen1:0')
            with open('test.txt', 'w') as f:
                f.write(str(a))
            d, np_img = sess.run([d, tensor_img])
            print(d[0][0])
            with open('img.txt', 'w') as f:
                f.write(str(np_img))
            imwrite("output_image_lowres.jpg", self.flip_channel_order(np_img, img_dim = 64))
