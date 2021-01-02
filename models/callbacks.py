import torch
import skorch
import torchvision


class TensorBoardWithImages(skorch.callbacks.TensorBoard):
    def on_batch_end(self, net, X, y, training, loss, y_pred):
        n_batches = len(net.history[-1, "batches"])
        if (n_batches % 10) != 0:
            return

        label_grid = torchvision.utils.make_grid(y.unsqueeze(1).float())
        self.writer.add_image("true_labels", label_grid, n_batches)

        probas = torch.nn.functional.softmax(y_pred)
        pred_grid = torchvision.utils.make_grid(probas.unsqueeze(1))
        self.writer.add_image("pred_labels", pred_grid, n_batches)
