import skorch
import torchvision


class TensorBoardWithImages(skorch.callbacks.TensorBoard):
    def on_batch_end(self, net, X, y, training, loss, y_pred):
        label_grid = torchvision.utils.make_grid(y.unsqueeze(1))
        self.writer.add_image("true_labels", label_grid)

        pred_grid = torchvision.utils.make_grid(y_pred.unsqueeze(1))
        self.writer.add_image("pred_labels", pred_grid)
