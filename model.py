import numpy as np
import tensorflow as tf
import encoder as enc
import preprocessing as pr

class StackGAN:
    def __init__(self, output_dim):
        self.output_dim = output_dim #Assumes we are attempting to generate a square image
        self.caption_arr = pr.construct_caption_arr(pr.num_to_attr('data/LabelledBirds/attributes/attributes.txt'), 'data/LabelledBirds/attributes/image_attribute_labels.txt')
    #Stage 1 GAN
    def generator_1(self, text_input, is_train = True):
        cond_vec = enc.encode().predict(enc.tokenize(text_input))
        noise_vec = tf.random_normal(cond_vec.shape)
        input = tf.concat([noise_vec, cond_vec], 1)
        with tf.variable_scope('gen1') as scope:
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
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act3 = tf.nn.relu(bn3)

            conv4 = tf.layers.conv2d_transpose(conv3, 3, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5), name = 'gen6') #Add filters,
            bn4 = tf.contrib.layers.batch_norm(conv4, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act4 = tf.nn.tanh(bn4)

            return tf.squeeze(act4) #should be a tensor of shape (output_dim, output_dim, 3)

    #Input is your image. Discriminator is only used during training, so you add the input when training
    def discriminator_1(self, input_img, input_txt, is_train = True):
        output_dim = 1
        init_embedding = enc.encode().predict(enc.tokenize(input_txt))
        #Shrink to lower dimension
        with tf.variable_scope('dis1', reuse = tf.AUTO_REUSE) as scope:
            #Reshape/Resize Text Embedding
            N, M = 75, 32 #Play around with these values. Likely, larger the value, more detail is retained but slower the training is
            W0 = tf.get_variable('dis11', shape = (init_embedding.shape[1], M*M*N), dtype = tf.float32, initializer = tf.truncated_normal_initializer)
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

            input_conv1 = tf.layers.conv2d(tf.to_float(resized_img), N, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'dis3')
            input_bn1 = tf.contrib.layers.batch_norm(input_conv1, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act1 = tf.nn.relu(input_bn1)

            #Concatenating img and text tensors, then preform convolutions
            input_concat = tf.concat([mod_embedding, tf.squeeze(input_act1)], 2)
            lst = [1]
            for i in input_concat.shape:
                if i is not None:
                    lst.append(i)
            input_concat = tf.reshape(input_concat, shape = lst)
            conv1 = tf.layers.conv2d(input_concat, 32, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02), name = 'conv1dis4')
            #Preform batch normalization because it was found to improve GAN preformance
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
            dim  = int(np.prod(act4.get_shape()[1:]))
            h = tf.reshape(act4, shape = [-1, dim])

            W1 = tf.get_variable(shape = [dim, 128], dtype = tf.float32, name = 'dis7')
            B1 = tf.get_variable(shape = [128], dtype = tf.float32, name = 'dis8')
            Y1 = tf.nn.relu(tf.add(tf.matmul(h, W1), B1))

            W2 = tf.get_variable(shape = [128, 1], dtype = tf.float32, name = 'dis9')
            B2 = tf.get_variable(shape = [1], dtype = tf.float32, name = 'dis10')
            Y2 = tf.add(tf.matmul(Y1, W2), B2)
            return tf.nn.sigmoid(Y2) #Use sigmoid activation because you want probability output


    def train_1(self):
        #print(enc.encode())
        #print(enc.encode().predict(enc.tokenize('hooked')))
        real_image_size = 64
        #graph1 = tf.Graph()

        # with graph1.as_default():
        text_input = tf.placeholder(dtype = tf.string)
        r_image = tf.placeholder(dtype = tf.string)

        real_image = 0
        t_i = 0
        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)
        try:
            sess = tf.InteractiveSession()
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
            sess = tf.InteractiveSession()
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
        dis_loss = tf.reduce_mean(real_result_fake_caption) + tf.reduce_mean(fake_result) - tf.reduce_mean(real_result_real_caption)
        gen_loss = -tf.reduce_mean(fake_result)

        #Creates lists of the discriminator and generator variables that will be trained
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'dis1' in var.name]
        g_vars = [var for var in t_vars if 'gen1' in var.name]

        print(len(d_vars))
        print(len(g_vars))

        trainer_gen = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(gen_loss, var_list = g_vars)
        trainer_dis = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(dis_loss, var_list = d_vars)

        with tf.Session() as sess:
            batch_size = 1
            num_of_imgs = 11788
            num_epochs = 1000 #adjust if necessary
            sess.run(tf.global_variables_initializer(), options = run_opts)
            sess.run(tf.local_variables_initializer(), options = run_opts)

            print('Start Training::: ')
            for i in range(num_epochs):
                print(str(i) + 'th epoch: ')
                feeder = pr.FeedExamples()
                num_of_batches = int(num_of_imgs/batch_size)
                for j in range(num_of_batches):
                    #Training the Discriminator.
                    for k in range(5):
                        train_data = feeder.next_example()
                        train_image = train_data[0]
                        #print(train_image)
                        txt = train_data[1]
                        #print(txt)
                        feed_txt = [[txt]]
                        _, dLoss = sess.run([dis_loss, trainer_dis], feed_dict = {text_input : feed_txt, r_image : train_image},
                                            options = run_opts)
                    #Training the Generator.
                    for k in range(1):
                        train_data = feeder.curr_example()
                        train_image = train_data[0]
                        txt = train_data[1]
                        _, gLoss = sess.run([gen_loss, trainer_gen], feed_dict = {text_input : [[txt]], r_image : train_image},
                                            options = run_opts)


                    print('Discriminator Loss: ' + str(dLoss))
                    print('Generator Loss: ' + str(gLoss))

    #Stage 2 GAN

    def generator_2(self, text_input, is_train = True):
        #Upsample text embedding
        init_embedding = enc.encode().predict(enc.tokenize(text_input))
        N, M = 75, 32 #Play around with these values. Likely, larger the value, more detail is retained but slower the training is
        with tf.variable_scope('gen2') as scope:
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

            conv_input = tf.concat([mod_embedding, tf.squeeze(input_act1)], 2)

            lst = [1]
            lst.extend(conv_input.shape)
            input = tf.reshape(conv_input, shape = lst)
            #Deconvolution to get high quality image
            conv1 = tf.layers.conv2d_transpose(input, 60, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5)) #Add filters,
            bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act1 = tf.nn.relu(bn1)

            conv2 = tf.layers.conv2d_transpose(act1, 30, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5)) #Add filters,
            bn2 = tf.contrib.layers.batch_norm(conv2, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act2 = tf.nn.relu(bn2)

            conv3 = tf.layers.conv2d_transpose(act2, 3, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = 0.5)) #Add filters,
            bn3 = tf.contrib.layers.batch_norm(conv3, is_training = is_train, epsilon = 1e-5, decay = 0.9, updates_collections = None)
            act3 = tf.nn.relu(bn3)

            return act3

    def discriminator_2(self, input_img, input_txt, is_train = True):
        output_dim = 1

        init_embedding = enc.encode().predict(enc.tokenize(input_txt)) #Refer back to blog post if this doesn't work

        with tf.variable_scope('dis2', reuse = tf.AUTO_REUSE) as scope:
            N, M = 75, 64 #Play around with these values. Likely, larger the value, more detail is retained but slower the training is
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

            input_conv1 = tf.layers.conv2d(tf.to_float(resized_img), int(N/2), kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02))
            input_bn1 = tf.contrib.layers.batch_norm(input_conv1, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act1 = tf.nn.relu(input_bn1)

            input_conv2 = tf.layers.conv2d(input_act1, N, kernel_size = [5, 5], strides = [2, 2], padding = 'SAME', kernel_initializer = tf.truncated_normal_initializer(stddev = .02))
            input_bn2 = tf.contrib.layers.batch_norm(input_conv2, is_training = is_train, epsilon=1e-5, decay = 0.9, updates_collections=None)
            input_act2 = tf.nn.relu(input_bn2)

            #Concatenating img and text tensors, then preform convolutions
            input_concat = tf.concat([mod_embedding, tf.squeeze(input_act2)], 2)

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



    def train_2(self):
        img_dim = 256
        text_input = tf.placeholder(tf.string)
        r_image = tf.placeholder(tf.string)

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
            sess = tf.InteractiveSession()
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
            sess = tf.InteractiveSession()
            img = tf.image.decode_jpeg(path, channels = 3)
            real_image = img.eval()
            sess.close()

        print('Generation/Discrimination 2 Start')
        fake_image = self.generator_2(t_i)
        print('First')
        real_result_real_caption = self.discriminator_2(real_image, t_i)
        print('second')
        real_result_fake_caption = self.discriminator_2(real_image, fake_caption)
        print('third')
        fake_result = self.discriminator_2(fake_image, t_i)
        print('fourth')

        dis_loss = tf.reduce_mean(real_result_fake_caption) + tf.reduce_mean(fake_result) - tf.reduce_mean(real_result_real_caption)
        gen_loss = -tf.reduce_mean(fake_result)

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'dis2' in var.name]
        g_vars = [var for var in t_vars if 'gen2' in var.name]

        trainer_dis = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(dis_loss, var_list = d_vars)
        trainer_gen = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(gen_loss, var_list = g_vars)

        with tf.Session() as sess:
            batch_size = 1
            num_of_imgs = 11788
            num_epochs = 1000 #adjust if necessary
            sess.run(tf.global_variables_initializer(), options = run_opts)
            sess.run(tf.local_variables_initializer(), options = run_opts)

            print('Start Training:::')
            for i in range(num_epochs):
                print(str(i) + 'th epoch: ')
                feeder = pr.FeedExamples()
                for j in range(int(num_of_imgs/batch_size)):
                    noise_vecs = np.random.uniform(-1.0, 1.0, size = (batch_size, input_dim))
                    #Training the Discriminator.
                    for k in range(5):
                        train_data = feeder.next_example()
                        train_image = train_data[0]
                        txt = train_data[1]
                        _, dLoss = sess.run([dis_loss, trainer_dis],
                                            feed_dict = {text_input : [[txt]], r_image : train_image},
                                            options = run_opts)
                    #Training the Generator.
                    for k in range(1):
                        train_data = feeder.curr_example()
                        train_image = train_data[0]
                        txt = train_data[1]
                        _, gLoss = sess.run([gen_loss, trainer_gen],
                                            feed_dict = {text_input : tf.constant([[txt]]), r_image : train_image},
                                                         options = run_opts)


                print('Discriminator Loss: ' + str(dLoss))
                print('Generator Loss: ' + str(gLoss))

    def predict(self, input_text):
        return tf.image.encode_jpeg(generator_2(input_text))

    def predict_lowres(self, input_text):
        return tf.image.encode_jpeg(generator_1(input_text))
