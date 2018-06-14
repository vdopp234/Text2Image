from model import StackGAN

model = StackGAN(256)

model.train_1()
model.train_2()

def predict(model, input_text, lowres = False):
    model.predict(input_text, is_train = False)
    if lowres:
        model.predict_lowres(input_text, is_train = False)
