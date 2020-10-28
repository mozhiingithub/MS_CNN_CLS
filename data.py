import numpy as np


class SampleLoader:
    def __init__(self):
        # ms = np.load('MS_NHWC_SPLIT.npz', allow_pickle=True)
        ms = np.load('MS_FINAL.npz', allow_pickle=True)
        x_trn: np.ndarray = ms['x_train']
        y_trn: np.ndarray = ms['y_train']
        x_val: np.ndarray = ms['x_val']
        y_val: np.ndarray = ms['y_val']
        x_tst: np.ndarray = ms['x_test']
        y_tst: np.ndarray = ms['y_test']

        # scale
        x_trn = x_trn.astype(np.float32) / 255
        x_val = x_val.astype(np.float32) / 255
        x_tst = x_tst.astype(np.float32) / 255

        y_trn = y_trn.astype(np.int32)
        y_val = y_val.astype(np.int32)
        y_tst = y_tst.astype(np.int32)

        trn_indices = np.arange(0, x_trn.shape[0])
        val_indices = np.arange(0, x_val.shape[0])

        self.x_trn = x_trn
        self.x_val = x_val
        self.x_tst = x_tst
        self.y_trn = y_trn
        self.y_val = y_val
        self.y_tst = y_tst

        self.trn_indices = trn_indices
        self.val_indices = val_indices

    def shuffle_indices(self):
        np.random.shuffle(self.trn_indices)
        np.random.shuffle(self.val_indices)

    def get_batch(self, start, end, val=False):
        if not val:
            indices = self.trn_indices
            x = self.x_trn
            y = self.y_trn
        else:
            indices = self.val_indices
            x = self.x_val
            y = self.y_val
        batch_indices = indices[start:end]
        return x[batch_indices], y[batch_indices]
