import numpy as np
import tensorflow as tf
import encoder as enc
import preprocessing as pr
from cv2 import imwrite
from tensorflow.python.framework import ops


class StackGAN:
    def __init__(self, output_dim):
        #self.model_session = tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)))
        self.output_dim = output_dim #Assumes we are attempting to generate a square image
        self.caption_arr = pr.construct_caption_arr(pr.num_to_attr('data/LabelledBirds/attributes/attributes.txt'), 'data/LabelledBirds/attributes/image_attribute_labels.txt')
        self.stage_1_vars = []
    #Stage 1 GAN
    def generator_1(self, text_input, is_train = True):

        with tf.variable_scope('g1', reuse = tf.AUTO_REUSE) as scope:
            encoder_ = enc.load_model()
            cond_vec = encoder_.predict(enc.tokenize(text_input))
            noise_vec = tf.random_normal(cond_vec.shape)
            input = tf.concat([noise_vec, cond_vec], 1)

            W1 = tf.get_variable('gen1', shape = (input.shape[1].value, 4*4*self.output_dim), dtype = tf.float32, initializer = tf.truncated_normal_initializer)
            B1 = tf.get_variable('gen2', shape = (4*4*self.output_dim), dtype = tf.float32, initializer = tf.constant_initializer(0.0))
            Y1 = tf.add(tf.matmul(input, W1), B1)
            act_ffn = tf.nn.relu(Y1)

            reshaped = tf.reshape(act_ffn, shape = [-1, 4, 4, self.output_dim])
            output_dim = self.output_dim / 4

            start = int(self.output_dim/2)

        #print(reshaped)
            conv1 = tf.layers.conv2d_transpose(reshaped, start, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5), name = 'gen3') #Add filters,
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act1 = tf.nn.relu(bn1)

            conv2 = tf.layers.conv2d_transpose(conv1, int(start/2), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5), name = 'gen4') #Add filters,
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act2 = tf.nn.relu(bn2)

            conv3 = tf.layers.conv2d_transpose(conv2, int(start/4), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5), name = 'gen5') #Add filters,
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None )
            act3 = tf.nn.relu(bn3)

            conv4 = tf.layers.conv2d_transpose(conv3, 3, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5), name = 'gen6') #Add filters,
            bn4 = tf.contrib.layers.batch_norm(conv4, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act4 = tf.nn.tanh(bn4)

            return tf.squeeze(act4) #should be a tensor of shape (output_dim, output_dim, 3)

    #Input is your image. Discriminator is only used during training, so you add the input when training
    def discriminator_1(self, input_img, input_txt, is_train = True):
        #Shrink to lower dimension
        with tf.variable_scope('d1', reuse = tf.AUTO_REUSE) as scope:
            encoder_ = enc.load_model()

            #Reshape/Resize Text Embedding
            output_dim = 1
            init_embedding = encoder_.predict(enc.tokenize(input_txt))
            N, M = 16, 32 #Play around with these values. Likely, larger the value, more detail is retained but slower the training is

            W0 = tf.get_variable('dis1', shape = (init_embedding.shape[1], M*M*N), dtype = tf.float32, initializer = tf.truncated_normal_initializer)
            B0 = tf.get_variable('dis2', shape = (M*M*N,), dtype = tf.float32, initializer = tf.constant_initializer(0.0))
            Y0 = tf.nn.relu(tf.add(tf.matmul(init_embedding, W0), B0))

                #Expand to create MxMxN 3d tensor
            mod_embedding = tf.reshape(Y0, shape = (M, M, N))

                #Convolution for img
            resized_img = tf.image.resize_images(input_img, (64, 64))
            lst = resized_img.shape
                #print(resized_img.shape) #Debugging Purposes
            if len(resized_img.shape) != 4:
                lst = [1]
                for i in resized_img.shape:
                    lst.append(i)
            resized_img = tf.reshape(resized_img, shape = lst)
                #print(input_img.dtype)

            input_conv1 = tf.layers.conv2d(tf.to_float(resized_img), int(N), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'dis3')
            input_bn1 = tf.contrib.layers.batch_norm(input_conv1, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act1 = tf.nn.relu(input_bn1)

                # input_conv2 = tf.layers.conv2d(input_act1, int(N/4), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'dis31')
                # input_bn2 = tf.contrib.layers.batch_norm(input_conv2, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
                # input_act2 = tf.nn.relu(input_bn2)
                #
                # input_conv3 = tf.layers.conv2d(input_act2, int(N/2), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'dis32')
                # input_bn3 = tf.contrib.layers.batch_norm(input_conv3, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
                # input_act3 = tf.nn.relu(input_bn3)
                #
                # input_conv4 = tf.layers.conv2d(input_act3, int(N), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'dis33')
                # input_bn4 = tf.contrib.layers.batch_norm(input_conv4, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
                # input_act4 = tf.nn.relu(input_bn4)

                #Concatenating img and text tensors, then preform convolutions
            input_concat = tf.concat([mod_embedding, tf.squeeze(input_act1)], 2)
            lst = [1]
            for i in input_concat.shape:
                if i is not None:
                    lst.append(i)
            input_concat = tf.reshape(input_concat, shape = lst)
            conv1 = tf.layers.conv2d(input_concat, 32, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv1dis4')
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            act1 = tf.nn.relu(bn1)

            conv2 = tf.layers.conv2d(act1, 64, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv2dis5')
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act2 = tf.nn.relu(bn2)

            conv3 = tf.layers.conv2d(act2, 128, kernel_size = [5, 5], strides = (2, 2), padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv3dis6')
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act3 = tf.nn.relu(bn3)

            conv4 = tf.layers.conv2d(act3, 256, kernel_size = [5, 5], strides = (2,2), padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv4dis7')
            bn4 = tf.contrib.layers.batch_norm(conv4, decay = 0.9, epsilon = 1e-5, is_training = is_train, updates_collections = None)
            act4 = tf.nn.relu(bn4)

                #Feed Forward Network
            dim  = int(np.prod(act1.get_shape()[1:]))
            h = tf.reshape(act1, shape = [-1, dim])

            W1 = tf.get_variable(shape = [dim, 128], dtype = tf.float32, name = 'dis7')
            B1 = tf.get_variable(shape = [128], dtype = tf.float32, name = 'dis8')
            Y1 = tf.nn.relu(tf.add(tf.matmul(h, W1), B1))

            W2 = tf.get_variable(shape = [128, 1], dtype = tf.float32, name = 'dis9')
            B2 = tf.get_variable(shape = [1], dtype = tf.float32, name = 'dis10')
            Y2 = tf.add(tf.matmul(Y1, W2), B2)
            return tf.nn.sigmoid(Y2)

    def train_1(self, save = True):

        real_image_size = 64
        text_input = tf.placeholder(dtype = tf.string)
        r_image = tf.placeholder(dtype = tf.string)
        lr = tf.placeholder(dtype = tf.float32)
        real_image = 0
        t_i = 0
        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        try:
            sess = tf.InteractiveSession(config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)))
            t_i = text_input.eval()
            t_i = text_input[0][0]
            path = tf.read_file(r_image)
            img = tf.image.decode_jpeg(path, channels = 3)
            real_image = img.eval()
            sess.close()
        except Exception:
            sess.close()
            t_i = 'seabird'
            path = 'data/LabelledBirds/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
            path = tf.read_file(path)
            sess = tf.InteractiveSession(config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)))
            img = tf.image.decode_jpeg(path, channels = 3)
            real_image = img.eval()
            sess.close()

        all_captions = self.caption_arr
        rand_idx = np.random.random()*11788
        fake_caption = all_captions[int(rand_idx)]
        while text_input == fake_caption:
            rand_idx = np.random.random()*len(all_captions)
            fake_caption = all_captions[int(rand_idx)]
        #All print statements are for debugging purposes only
        print('Generation/Discrimination Start')
        fake_image_size = 64
        fake_image = self.generator_1(t_i)
        print('First')

        fake_result = self.discriminator_1(fake_image, t_i)
        print('Second')
        real_result_fake_caption = self.discriminator_1(real_image, fake_caption)
        print('Third')
        real_result_real_caption = self.discriminator_1(real_image, t_i)

        print('Finished 3 discriminations')
        RRRC = tf.reduce_mean(real_result_real_caption)
        RRFC = tf.reduce_mean(real_result_fake_caption)
        FR = tf.reduce_mean(fake_result)

        dis_loss = -1*(tf.log(RRRC) + tf.log(1 - FR) + tf.log(1 - RRFC))
        #dis_loss = tf.log(1 - RRRC) + tf.log(FR) + tf.log(RRFC)
        gen_loss = tf.log(1 - FR)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd1' in var.name]
        g_vars = [var for var in t_vars if 'g1' in var.name]

        #with tf.variable_scope('gan1', reuse = tf.AUTO_REUSE):
        trainer_gen = tf.train.AdamOptimizer(learning_rate = lr, name = 'stage1gen').minimize(gen_loss, var_list = g_vars)
        trainer_dis = tf.train.AdamOptimizer(learning_rate = lr, name = 'stage1dis').minimize(dis_loss, var_list = d_vars)



        with tf.Session() as sess:
            batch_size = 10
            num_of_imgs = 11788
            num_epochs = 2 #adjust if necessary

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
                avg_loss = 0
                for j in range(num_of_batches):
                    #Training the Generator.
                    ind = int(np.random.random()*5)
                    example = None
                    for _ in range(10):
                        train_data = feeder.next_example()
                        if _ == ind:
                            example = train_data
                        train_image = train_data[0]
                        txt = train_data[1]
                        l_r = 1e-4 * (0.5 ** (i//100))
                        gLoss, op = sess.run([gen_loss, trainer_gen], feed_dict = {text_input : [[txt]], r_image : train_image, lr : l_r},
                                                options = run_opts)
                    for _ in range(1):
                        train_image = example[0]
                        txt = example[1]
                        l_r = 8e-4 * (0.5 ** (i//100))
                        dLoss, FR_dis, op = sess.run([dis_loss, FR, trainer_dis], feed_dict = {text_input : [[txt]], r_image : train_image, lr : l_r},
                                                options = run_opts)

                    print('Generator Loss: ' + str(gLoss))
                    print('Discriminator Loss: ' + str(dLoss))
                    print('Fake Image: ' + str(FR_dis))
                #Save current state after every epoch
                if save:
                    print('Saving:')
                    saver = tf.train.Saver()
                    saver.save(sess, "./ckpts/model.ckpt")
    #Stage 2 GAN

    def generator_2(self, text_input, is_train = True, r = tf.AUTO_REUSE):
        #Upsample text embedding
        #Play around with these values. Likely, larger the value, more detail is retained but slower the training is
        encoder_ = enc.load_model()
        with tf.variable_scope('gen2', reuse = r) as scope:
            init_embedding = encoder_.predict(enc.tokenize(text_input))
            N, M = 32, 16

            W0 = tf.get_variable('w0', shape = (init_embedding.shape[1], M*M*N), dtype = tf.float32, initializer = tf.truncated_normal_initializer)
            B0 = tf.get_variable('b0', shape = (M*M*N,), dtype = tf.float32, initializer = tf.constant_initializer(0.0))
            Y0 = tf.nn.relu(tf.add(tf.matmul(init_embedding, W0), B0))

            #Expand to create MxMxN 3d tensor
            mod_embedding = tf.reshape(Y0, shape = (M, M, N))

            #Downsample gan1 image
            gan1img = self.generator_1(text_input)
            lst = [1]
            lst.extend(gan1img.shape)
            input_img = tf.reshape(gan1img, shape = lst)

            input_conv1 = tf.layers.conv2d(input_img, N, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02))
            input_bn1 = tf.contrib.layers.batch_norm(input_conv1, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act1 = tf.nn.relu(input_bn1)

            input_conv2 = tf.layers.conv2d(input_act1, N, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02))
            input_bn2 = tf.contrib.layers.batch_norm(input_conv2, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act2 = tf.nn.relu(input_bn2)

            conv_input = tf.concat([mod_embedding, tf.squeeze(input_act2)], 2)

            lst = [1]
            lst.extend(conv_input.shape)
            input = tf.reshape(conv_input, shape = lst)
            #Deconvolution to get high quality image
            conv1 = tf.layers.conv2d_transpose(input, 60, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5)) #Add filters,
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act1 = tf.nn.relu(bn1)

            conv2 = tf.layers.conv2d_transpose(act1, 40, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5)) #Add filters,
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act2 = tf.nn.relu(bn2)

            conv3 = tf.layers.conv2d_transpose(act2, 20, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5)) #Add filters,
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act3 = tf.nn.relu(bn3)

            conv4 = tf.layers.conv2d_transpose(act3, 3, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5)) #Add filters,
            bn4 = tf.contrib.layers.batch_norm(conv4, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act4 = tf.nn.relu(bn4)

            return act4

    def discriminator_2(self, input_img, input_txt, is_train = True, r = tf.AUTO_REUSE):
        with tf.variable_scope('dis2', reuse = r) as scope:
            output_dim = 1
            init_embedding = enc.load_model().predict(enc.tokenize(input_txt)) #Refer back to blog post if this doesn't work
            N, M = 16, 16 #Play around with these values. Likely, larger the value, more detail is retained but OOM Error may be thrown.
            W0 = tf.get_variable('w0', shape = (init_embedding.shape[1], M*M*N), dtype = tf.float32, initializer = tf.truncated_normal_initializer)
            B0 = tf.get_variable('b0', shape = (M*M*N,), dtype = tf.float32, initializer = tf.constant_initializer(0.0))
            Y0 = tf.nn.relu(tf.add(tf.matmul(init_embedding, W0), B0))

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
            input_act1 = tf.nn.relu(input_bn1)

            input_conv2 = tf.layers.conv2d(input_act1, int(N/2), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02))
            input_bn2 = tf.contrib.layers.batch_norm(input_conv2, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act2 = tf.nn.relu(input_bn2)

            input_conv3 = tf.layers.conv2d(input_act2, 3*int(N/4), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02))
            input_bn3 = tf.contrib.layers.batch_norm(input_conv3, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act3 = tf.nn.relu(input_bn3)

            input_conv4 = tf.layers.conv2d(input_act3, N, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02))
            input_bn4 = tf.contrib.layers.batch_norm(input_conv4, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act4 = tf.nn.relu(input_bn4)
            #Concatenating img and text tensors, then preform convolutions
            input_concat = tf.concat([mod_embedding, tf.squeeze(input_act4)], 2)

            lst = [1]
            lst.extend(input_concat.shape)
            img = tf.reshape(input_concat, shape = lst)
            conv1 = tf.layers.conv2d(img, 128, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv1')
            #Preform batch normalization because it was found to improve GAN preformance
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            act1 = tf.nn.relu(bn1)

            conv2 = tf.layers.conv2d(act1, 256, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv2')
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act2 = tf.nn.relu(bn2)

            conv3 = tf.layers.conv2d(act2, 512, kernel_size = [5, 5], strides = (2, 2), padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv3')
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act3 = tf.nn.relu(bn3)

            # conv4 = tf.layers.conv2D(act3, 256, kernel_size = [5, 5], strides = (2,2), padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv4')
            # bn4 = tf.contrib.layers.batch_norm(conv4, decay = 0.9, epsilon = 1e-5, is_training = is_train, updates_collections = None)
            # act4 = tf.nn.relu(bn4)

            #Feed Forward Network
            dim  = int(np.prod(act3.get_shape()[1:]))#1: because of batch size
            h = tf.reshape(act3, shape = [-1, dim])

            W1 = tf.get_variable('dis2w1', shape = [dim, 128], dtype = tf.float32)
            B1 = tf.get_variable('dis2b1', shape = [128], dtype = tf.float32)
            Y1 = tf.nn.relu(tf.add(tf.matmul(h, W1), B1))

            W2 = tf.get_variable('dis2w2', shape = [128, 1], dtype = tf.float32)
            B2 = tf.get_variable('dis2b2', shape = [1], dtype = tf.float32)
            Y2 = tf.add(tf.matmul(Y1, W2), B2) #Use sigmoid activation because you want probability output
            return tf.nn.sigmoid(Y2)
        #Shrink to lower dimension



    def train_2(self, save = True):
        print('Finished Training Gen 1')
        img_dim = 256
        text_input = tf.placeholder(tf.string)
        r_image = tf.placeholder(tf.string)
        lr = tf.placeholder(tf.float32)

        all_captions = self.caption_arr
        rand_idx = np.random.random()*len(all_captions)
        fake_caption = all_captions[int(rand_idx)]
        while text_input == fake_caption:
            rand_idx = np.random.random()*len(all_captions)
            fake_caption = all_captions[int(rand_idx)]

        t_i = ''
        real_image = 0
        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        try:
            sess = tf.InteractiveSession(config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)))
            t_i = text_input.eval()
            t_i = t_i[0][0]
            path = tf.read_file(r_image)
            img = tf.image.decode_jpeg(path, channels = 3)
            real_image = img.eval()
            sess.close()
        except Exception:
            sess.close()
            t_i = 'seabird'
            path = 'data/LabelledBirds/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg'
            path = tf.read_file(path)
            sess = tf.InteractiveSession(config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)))
            img = tf.image.decode_jpeg(path, channels = 3)
            real_image = img.eval()
            sess.close()

        print('Generation/Discrimination 2 Start')
        fake_image = self.generator_2(t_i, is_train = False)
        print('First')
        real_result_real_caption = self.discriminator_2(real_image, t_i)
        print('second')
        real_result_fake_caption = self.discriminator_2(real_image, fake_caption)
        print('third')
        fake_result = self.discriminator_2(fake_image, t_i)
        print('fourth')

        RRFC = tf.reduce_mean(real_result_fake_caption)
        FR = tf.reduce_mean(fake_result)
        RRRC = tf.reduce_mean(real_result_real_caption)

        dis_loss = -1*(tf.log(RRRC) + tf.log(1 - FR) + tf.log(1 - RRFC))
        #dis_loss_1 = tf.reduce_mean(real_result_fake_caption) + tf.reduce_mean(fake_result) - tf.reduce_mean(real_result_real_caption)
        gen_loss = tf.log(1 - FR)
        #gen_loss_1 = -tf.reduce_mean(fake_result)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'dis2/' in var.name]
        g_vars = [var for var in t_vars if 'gen2/' in var.name]

        with tf.variable_scope('gan2', reuse = tf.AUTO_REUSE):
            trainer_dis = tf.train.AdamOptimizer(learning_rate = lr).minimize(dis_loss, var_list = d_vars)
            trainer_gen = tf.train.AdamOptimizer(learning_rate = lr).minimize(gen_loss, var_list = g_vars)

        #Train 2
        with tf.Session() as sess:
            batch_size = 1
            num_of_imgs = 11788
            num_epochs = 1000 #adjust if necessary

            #Load Stage1 GAN Graph
            saver = tf.train.import_meta_graph('ckpts/model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('ckpts'))

            #Defines Variables that should be initialized for training
            global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
            self.stage_1_vars = ['embedding_1/embeddings:0', 'gen1/gen1:0', 'gen1/gen2:0', 'gen1/gen3/kernel:0', 'gen1/gen3/bias:0', 'gen1/BatchNorm/beta:0', 'gen1/BatchNorm/moving_mean:0', 'gen1/BatchNorm/moving_variance:0', 'gen1/gen4/kernel:0', 'gen1/gen4/bias:0', 'gen1/BatchNorm_1/beta:0', 'gen1/BatchNorm_1/moving_mean:0', 'gen1/BatchNorm_1/moving_variance:0', 'gen1/gen5/kernel:0', 'gen1/gen5/bias:0', 'gen1/BatchNorm_2/beta:0', 'gen1/BatchNorm_2/moving_mean:0', 'gen1/BatchNorm_2/moving_variance:0', 'gen1/gen6/kernel:0', 'gen1/gen6/bias:0', 'gen1/BatchNorm_3/beta:0', 'gen1/BatchNorm_3/moving_mean:0', 'gen1/BatchNorm_3/moving_variance:0', 'embedding_2/embeddings:0', 'dis1/dis11:0', 'dis1/dis2:0', 'dis1/dis3/kernel:0', 'dis1/dis3/bias:0', 'dis1/BatchNorm/beta:0', 'dis1/BatchNorm/moving_mean:0', 'dis1/BatchNorm/moving_variance:0', 'dis1/conv1dis4/kernel:0', 'dis1/conv1dis4/bias:0', 'dis1/BatchNorm_1/beta:0', 'dis1/BatchNorm_1/moving_mean:0', 'dis1/BatchNorm_1/moving_variance:0', 'dis1/conv2dis5/kernel:0', 'dis1/conv2dis5/bias:0', 'dis1/BatchNorm_2/beta:0', 'dis1/BatchNorm_2/moving_mean:0', 'dis1/BatchNorm_2/moving_variance:0', 'dis1/conv3dis6/kernel:0', 'dis1/conv3dis6/bias:0', 'dis1/BatchNorm_3/beta:0', 'dis1/BatchNorm_3/moving_mean:0', 'dis1/BatchNorm_3/moving_variance:0', 'dis1/conv4dis7/kernel:0', 'dis1/conv4dis7/bias:0', 'dis1/BatchNorm_4/beta:0', 'dis1/BatchNorm_4/moving_mean:0', 'dis1/BatchNorm_4/moving_variance:0', 'dis1/dis7:0', 'dis1/dis8:0', 'dis1/dis9:0', 'dis1/dis10:0', 'embedding_3/embeddings:0', 'embedding_4/embeddings:0', 'beta1_power:0', 'beta2_power:0', 'gen1/gen1/Adam:0', 'gen1/gen1/Adam_1:0', 'gen1/gen2/Adam:0', 'gen1/gen2/Adam_1:0', 'gen1/gen3/kernel/Adam:0', 'gen1/gen3/kernel/Adam_1:0', 'gen1/gen3/bias/Adam:0', 'gen1/gen3/bias/Adam_1:0', 'gen1/gen4/kernel/Adam:0', 'gen1/gen4/kernel/Adam_1:0', 'gen1/gen4/bias/Adam:0', 'gen1/gen4/bias/Adam_1:0', 'gen1/gen5/kernel/Adam:0', 'gen1/gen5/kernel/Adam_1:0', 'gen1/gen5/bias/Adam:0', 'gen1/gen5/bias/Adam_1:0', 'gen1/gen6/kernel/Adam:0', 'gen1/gen6/kernel/Adam_1:0', 'gen1/gen6/bias/Adam:0', 'gen1/gen6/bias/Adam_1:0', 'gen1/BatchNorm_3/beta/Adam:0', 'gen1/BatchNorm_3/beta/Adam_1:0', 'beta1_power_1:0', 'beta2_power_1:0', 'dis1/dis11/Adam:0', 'dis1/dis11/Adam_1:0', 'dis1/dis2/Adam:0', 'dis1/dis2/Adam_1:0', 'dis1/dis3/kernel/Adam:0', 'dis1/dis3/kernel/Adam_1:0', 'dis1/dis3/bias/Adam:0', 'dis1/dis3/bias/Adam_1:0', 'dis1/BatchNorm/beta/Adam:0', 'dis1/BatchNorm/beta/Adam_1:0', 'dis1/conv1dis4/kernel/Adam:0', 'dis1/conv1dis4/kernel/Adam_1:0', 'dis1/conv1dis4/bias/Adam:0', 'dis1/conv1dis4/bias/Adam_1:0', 'dis1/BatchNorm_1/beta/Adam:0', 'dis1/BatchNorm_1/beta/Adam_1:0', 'dis1/conv2dis5/kernel/Adam:0', 'dis1/conv2dis5/kernel/Adam_1:0', 'dis1/conv2dis5/bias/Adam:0', 'dis1/conv2dis5/bias/Adam_1:0', 'dis1/BatchNorm_2/beta/Adam:0', 'dis1/BatchNorm_2/beta/Adam_1:0', 'dis1/conv3dis6/kernel/Adam:0', 'dis1/conv3dis6/kernel/Adam_1:0', 'dis1/conv3dis6/bias/Adam:0', 'dis1/conv3dis6/bias/Adam_1:0', 'dis1/BatchNorm_3/beta/Adam:0', 'dis1/BatchNorm_3/beta/Adam_1:0', 'dis1/conv4dis7/kernel/Adam:0', 'dis1/conv4dis7/kernel/Adam_1:0', 'dis1/conv4dis7/bias/Adam:0', 'dis1/conv4dis7/bias/Adam_1:0', 'dis1/BatchNorm_4/beta/Adam:0', 'dis1/BatchNorm_4/beta/Adam_1:0', 'dis1/dis7/Adam:0', 'dis1/dis7/Adam_1:0', 'dis1/dis8/Adam:0', 'dis1/dis8/Adam_1:0', 'dis1/dis9/Adam:0', 'dis1/dis9/Adam_1:0', 'dis1/dis10/Adam:0', 'dis1/dis10/Adam_1:0']
            init_vars = []
            init_var_names = []
            for var in global_vars:
                if 'embedding_1/embeddings' in var.name:
                    print(var)
                    print(1)
                if var.name not in self.stage_1_vars and var.name not in init_var_names and type(var) == tf.Variable:
                    if 'embedding_1/embeddings' in var.name:
                        print(var)
                        print(2)
                    init_vars.append(var)
                    init_var_names.append(var.name)
            # print(len(init_vars))
            # [var.name for var in init_vars]
            sess.run(tf.variables_initializer(var_list = init_vars), options = run_opts)
            print('Start Training:::')
            for i in range(num_epochs):
                print(str(i) + 'th epoch: ')
                feeder = pr.FeedExamples()
                for j in range(int(num_of_imgs/batch_size)):
                    #noise_vecs = np.random.uniform(-1.0, 1.0, size = (batch_size, input_dim))
                    for _ in range(batch_size):
                        #Training the Generator.
                        for k in range(1):
                            train_data = feeder.next_example()
                            train_image = train_data[0]
                            txt = train_data[1]
                            l_r = 3e-3
                            gloss, RRFC_gen, FR_gen, RRRC_gen, _ = sess.run([gen_loss, RRFC, FR, RRRC, trainer_gen],
                                                feed_dict = {text_input : [[txt]], r_image : train_image, lr : l_r},
                                                             options = run_opts)
                            dLoss, FR_dis, _ = sess.run([dis_loss, FR, trainer_dis],
                                                feed_dict = {text_input : [[txt]], r_image : train_image, lr: l_r},
                                                options = run_opts)
                            print('Discriminator Loss: ' + str(dLoss))
                            print('Generator Loss: ' + str(gLoss))
                if save:
                    saver_1 = tf.train.Saver()
                    saver_1.save(sess, "./ckpts/model.ckpt")

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
            saver = tf.train.import_meta_graph('ckpts/model.ckpt.meta')
            saver.restore(sess, tf.train.latest_checkpoint('ckpts'))
            # names = []
            # for v in tf.get_default_graph().get_collection('variables'):
            #     names.append(v.name)
            # print(names)
            d, np_img = sess.run([d, tensor_img])
            print(d)
            imwrite("output_image.jpg", self.flip_channel_order(np_img))

    def predict_lowres(self, input_text, exists = False):
        tf.reset_default_graph()
        init_img = self.generator_1(input_text)
        d = self.discriminator_1(init_img, input_text, is_train = True)
        tensor_img = tf.squeeze(init_img)
        with tf.Session() as sess:

            saver = tf.train.Saver()
            #saver = tf.train.import_meta_graph('ckpts/model.ckpt.meta')
            #print(len(tf.get_default_graph().get_collection('variables')))
            saver.restore(sess, tf.train.latest_checkpoint('ckpts'))
            #print(len(tf.get_default_graph().get_collection('variables')))

            #print(len(sess.run(tf.report_uninitialized_variables())))

            a = sess.run('g1/gen1:0')
            with open('test.txt', 'w') as f:
                f.write(str(a))
            # if exists:
            #     old = ''
            #     lst = []
            #     with open('test.txt') as f:
            #         old = f.read().split(',')
            #     for i in range(len(a)):
            #         for j in range(len(a[0])):
            #             lst.append(a[i][j])
            #     for i in range(len(lst)):
            #         if lst[i] != old[i]:
            #             print(str(lst[i]) + ': ' + str(old[i]))
            #             print('Different!!!!!')
            #
            # for i in range(len(a)):
            #     for j in range(len(a[0])):
            #         with open('test.txt', 'w') as f:
            #             f.write(str(a[i][j]) + ',')

            # for var in l1:
            #     if var not in l2:
            #         print(var)
            # init_img = self.generator_1(input_text, tf.variable_scope('gen1', reuse = tf.AUTO_REUSE))
            # tensor_img = tf.squeeze(tf.cast(init_img, dtype = tf.uint8))
            # print(len(sess.run(tf.report_uninitialized_variables())))
            d, np_img = sess.run([d, tensor_img])
            print(d[0][0])
            with open('img.txt', 'w') as f:
                f.write(str(np_img))
            imwrite("output_image_lowres.jpg", self.flip_channel_order(np_img, img_dim = 64))
