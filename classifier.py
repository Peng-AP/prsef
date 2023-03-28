import tensorflow as tf
from tensorflow import keras

class single_predictor:
    def __init__(self,*args):
        if(len(args) == 2):self.predictor = keras.models.load_model(f"C:\\Users\\AaronPeng\\Desktop\\PRSEF\\sliding_window\\{args[1]}models\\{args[0]}")
        else: self.predictor = keras.models.load_model(args[0])
    def predict(self, imgs):
        result = self.predictor.predict(imgs)
        return result
    def summary(self): self.predictor.summary()