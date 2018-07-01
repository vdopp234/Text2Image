from model import StackGAN

model = StackGAN(256)

#model.train_1()
#model.train_2()

#For debugging purposes
test_input_text = 'has bill shape hooked seabird  has head pattern masked  has throat color buff  has eye color brown  has bill length longer than head  has forehead color white  has nape color buff  has primary color buff  has bill color buff  has crown color buff'

#model.predict(test_input_text)

def predict(model, input_text, lowres = False):
    model.predict(input_text)
    if lowres:
        model.predict_lowres(input_text)
