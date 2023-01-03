import tensorflow as tf
from tensorflow import keras

class single_predictor:
    def __init__(self,version):
        self.predictor = keras.models.load_model(f"C:\\Users\\AaronPeng\\Desktop\\PRSEF\\sliding_window\\models\\{version}")
    def predict(self, imgs):
        result = self.predictor.predict(imgs)
        return result
    def summary(self): self.predictor.summary()