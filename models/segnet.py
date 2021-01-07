import torch
import gcsfs
import skorch
import numpy as np
import contextlib


from models.metrics import dice


@contextlib.contextmanager
def _fs(fs, path, code="wb"):
    if isinstance(path, str) and path.startswith("gs://"):
        with fs.open(path, code) as f:
            yield f
    else:
        yield path


class SegNet(skorch.NeuralNet):
    def predict_proba(self, X, y=None):
        logits = super().predict_proba(X)
        return 1 / (1 + np.exp(-logits))

    def save_params(self, **kwargs):
        fs = gcsfs.GCSFileSystem()
        with contextlib.ExitStack() as stack:
            nkwargs = {
                par_name: stack.enter_context(_fs(fs, par_value))
                for par_name, par_value in kwargs.items()
            }
            super().save_params(**nkwargs)

    def thresholds(self, X, func=dice):
        dataset = self.get_dataset(X)
        dices = []
        for X, mask in self.get_iterator(dataset, training=False):
            logits = self.evaluation_step(X, training=False)
            yproba = torch.sigmoid(logits)

            batch = func(yproba.cpu(), mask.cpu())
            dices.append(batch.mean(axis=(1, 2)))

        return dices
