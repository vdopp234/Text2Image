#Keras Imports
from keras.layers import Embedding, Flatten
from keras.models import Sequential, model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#Preprocessing imports/variables
import preprocessing as pr
import glove
import pickle
vocab_size = 104
input_size = 70 #If caption word count is less than input_size, zero-pad. If a caption has word count >20, increase input_size

def get_tokenizer():
    tok = glove.training_data(pr.construct_caption_arr(pr.num_to_attr('data/LabelledBirds/attributes/attributes.txt'), 'data/LabelledBirds/attributes/image_attribute_labels.txt'))
    with open('saved_models/tok.pickle', 'wb') as handle:
        pickle.dump(tok, handle, protocol = pickle.HIGHEST_PROTOCOL)

def load_tokenizer():
    tok = None
    with open('saved_models/tok.pickle', 'rb') as handle:
        tok = pickle.load(handle)
    return tok

#Method for tokenizing and creating a proper vector from our input text that can be inputted into our NN
def tokenize(input_text):
    tok = load_tokenizer()
    a = tok.texts_to_sequences([input_text])
    return pad_sequences(a, input_size, padding = 'post') #Zeropads input vector to input_size. If greater than input_size, cuts off input

def get_model():
    embeddings = glove.load_glove_embeddings()
    model = Sequential()
    #Added an Embedding Layer with our glove embeddings for the words in our training dataself.
    #Not trainable because we don't want to update these weights. They are final
    model.add(Embedding(vocab_size, output_dim = 100, input_length = input_size, weights = [embeddings], trainable = False))
    model.add(Flatten())
    model_json = model.to_json()
    with open('saved_models/glove_encoder.json', 'w') as f:
        f.write(model_json)
    model.save_weights('saved_models/glove_weights.h5')
    return model

def load_model():
    model = open('saved_models/glove_encoder.json')
    loaded_model = model.read()
    model.close()
    loaded_model = model_from_json(loaded_model)
    loaded_model.load_weights('saved_models/glove_weights.h5')

    return loaded_model
