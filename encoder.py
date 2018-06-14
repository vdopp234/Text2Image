#Keras Imports
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#Preprocessing imports/variables
import preprocessing as pr
import glove
vocab_size = 104
input_size = 70 #If caption word count is less than input_size, zero-pad. If a caption has word count >20, increase input_size

#Method for tokenizing and creating a proper vector from our input text that can be inputted into our NN
def tokenize(input_text):
    tok = glove.training_data(pr.construct_caption_arr(pr.num_to_attr('data/LabelledBirds/attributes/attributes.txt'), 'data/LabelledBirds/attributes/image_attribute_labels.txt'))
    a = tok.texts_to_sequences([input_text])
    return pad_sequences(a, input_size, padding = 'post') #Zeropads input vector to input_size. If greater than input_size, cuts off input

#Architecture: Encoder 'reads' the input text and encodes it into an
#internal representation using pre-trained GLoVe weights.
def encode():
    embeddings = glove.load_glove_embeddings()
    model = Sequential()
    #Added an Embedding Layer with our glove embeddings for the words in our training dataself.
    #Not trainable because we don't want to update these weights. They are final
    model.add(Embedding(vocab_size, output_dim = 100, input_length = input_size, weights = [embeddings], trainable = False))
    model.add(Flatten())
    return model
