import os
import numpy as np
import tensorflow as tf


def num_to_attr(file_path):
    out = {}
    with open(file_path) as file:
        lst = file.readlines()
        for i in range(len(lst)):
            curr_str = lst[i]
            curr_lst = curr_str.split(' ')
            num = int(curr_lst[0])
            curr_str = curr_lst[1]
            curr_lst = curr_str.split('::')
            h = ''
            s = curr_lst[1].split('\n')[0]
            h1 = ''
            for i in s.split('_'):
                if '(' in i:
                    break
                h1 += i
            for i in curr_lst[0].split('_'):
                h += i + ' '
            curr_atr = h + h1
            out[num] = curr_atr
    return out
#Go over this method again and its uses in encoder.py and model.py
def construct_caption_arr(mapping, file_path):
    out = {}
    with open(file_path) as file:
        lst = file.readlines()
        num_imgs = 11788
        for i in range(num_imgs):
            caption = ""
            id = 1
            for j in range(312):
                #Grabs the correct String
                curr_str = lst[312 * i + j]
                #Splits attributes properly
                x = curr_str.split(' ')
                #Builds caption
                if int(x[2]) == 1 and int(x[3]) >= 3:
                    a = mapping[int(x[1])]
                    hold = ''
                    for word in a.split(' '):
                        #print(word)
                        word = help_split(word)
                        hold += word + ' '
                    caption += hold + ' '
                id = int(x[0])
            out[id] = caption
    return out

def help_split(word):
    if word == 'shorterthanhead':
        return 'shorter_than_head'
    elif word == 'aboutthesameashead':
        return 'about_the_same_as_head'
    elif word == 'notchedtail':
        return 'notched tail'
    elif word == 'verysmall':
        return 'very small'
    elif word == 'pointedtail':
        return 'pointed tail'
    elif word == 'eyering':
        return 'eye ring'
    elif word == 'eyeline':
        return 'eye line'
    elif word == 'roundedtail':
        return 'rounded tail'
    elif 'squaredtail' in word:
        return 'squared tail'
    elif word == 'shapedtail':
        return 'shaped tail'
    elif word == 'hookedseabird':
        return 'hooked seabird'
    elif word == 'longerthanhead':
        return 'longer than head'
    elif word == 'uniquepattern':
        return 'unique pattern'
    elif word == 'forkedtail':
        return 'forked tail'
    elif 'perchingwater' in word:
        return 'perching water'
    elif word == 'verylarge':
        return 'very large'
    else:
        return word

def img_to_id(file_path):
    out = {}
    with open(file_path) as file:
        lst = file.readlines()
        for i in lst:
            one = i.split(' ')
            id = int(one[0])
            two = one[1].split('\n')
            out[two[0]] = id
    return out

def is_num(str):
    try:
        int(str)
    except ValueError:
        return False
    return True

class FeedExamples:
    def __init__(self):
        self.curr_dir = 'data/LabelledBirds/images'
        self.curr_dir_num = 0
        self.curr_num_in_dir = -1
        self.caption_map = construct_caption_arr(num_to_attr('data/LabelledBirds/attributes/attributes.txt'), 'data/LabelledBirds/attributes/image_attribute_labels.txt')
        self.dir_lst = os.listdir(path = self.curr_dir)
        self.img_to_id = img_to_id('data/LabelledBirds/images.txt')

    def next_batch(self, batch_size, encode, img_size):
        ret_caps = None
        ret_imgs = None
        lst = [self.next_example() for _ in range(batch_size)]
        imgs = [tf.reshape(tf.image.resize_images(tf.cast(tf.image.decode_jpeg(tf.read_file(x[0]), channels = 3), dtype = tf.float32), (img_size, img_size)), shape = (1, 64, 64, 3)) for x in lst]
        caps = [encode(x[1]) for x in lst]
        h = tf.concat(imgs, axis = 0)
        ret_caps = np.concatenate(caps, axis = 0)
        sess = tf.InteractiveSession()
        ret_imgs = h.eval()
        sess.close()
        return ret_imgs, ret_caps

    def next_example(self):
        #Change to next directory
        # if self.curr_num_in_dir >= len(os.listdir(path = self.curr_dir + '/' + self.dir_lst[self.curr_dir_num])):
        #     self.curr_dir_num += 1
        #     self.curr_num_in_dir = -1
        #
        curr_dir = self.dir_lst[self.curr_dir_num]
        self.curr_num_in_dir += 1
        if self.curr_num_in_dir >= len(os.listdir(path = self.curr_dir + '/' + self.dir_lst[self.curr_dir_num])):
            self.curr_dir_num += 1
            self.curr_num_in_dir = -1
        img = os.listdir(path = self.curr_dir + '/' + curr_dir)[self.curr_num_in_dir]
        #img = tf.image.decode_jpeg(os.listdir(path = self.curr_dir + '/' + curr_dir)[self.curr_num_in_dir], channels = 3)
        caption_id = self.img_to_id[curr_dir + '/' + os.listdir(path = self.curr_dir + '/' + curr_dir)[self.curr_num_in_dir]]
        return (self.curr_dir + '/' + curr_dir + '/' + img, self.caption_map[caption_id])
    def curr_example(self):
        curr_dir = self.dir_lst[self.curr_dir_num]
        img = os.listdir(path = self.curr_dir + '/' + curr_dir)[self.curr_num_in_dir]
        #img = tf.image.decode_jpeg(os.listdir(path = self.curr_dir + '/' + curr_dir)[self.curr_num_in_dir], channels = 3)
        caption_id = self.img_to_id[curr_dir + '/' + os.listdir(path = self.curr_dir + '/' + curr_dir)[self.curr_num_in_dir]]
        return self.curr_dir + '/' + curr_dir + '/' + img, self.caption_map[caption_id]
