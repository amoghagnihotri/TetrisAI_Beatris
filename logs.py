from keras.callbacks import TensorBoard
from tensorboardX import FileWriter

class CustomTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = FileWriter(self.log_dir)

    def set_model(self, model):
        pass

    def log(self, step, **stats):
        # self.write(stats, step)
        pass
