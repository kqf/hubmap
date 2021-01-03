import torch
import skorch
import torchvision


class TensorBoardWithImages(skorch.callbacks.TensorBoard):
    def on_batch_end(self, net, X, y, training, loss, y_pred):
        global_step = sum(len(epoch["batches"]) for epoch in net.history)

        if (global_step % 10) != 0:
            return

        label_grid = torchvision.utils.make_grid(y.unsqueeze(1).float())
        self.writer.add_image("true_labels", label_grid, global_step)

        probas = torch.sigmoid(y_pred)
        pred_grid = torchvision.utils.make_grid(probas.unsqueeze(1))
        self.writer.add_image("pred_labels", pred_grid, global_step)
