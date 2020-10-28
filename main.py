import tensorflow as tf
import data
import model as m
import class_balanced_loss as cb
import datetime
import numpy as np
import os

epoch = 20
batch_size = 256
learning_rate = 1e-3

l2 = 0.1

training = True  # False for test.

cb_flag = True

init_cb_n = True

retrain_flag = False


def log(string):
    print(datetime.datetime.now(), string)


def main():
    sample_loader = data.SampleLoader()

    batch_num = int(sample_loader.trn_indices.shape[0] / batch_size)

    class DecayLr(tf.keras.optimizers.schedules.LearningRateSchedule):
        def get_config(self):
            pass

        def __init__(self):
            super(DecayLr, self).__init__()

        def __call__(self, step):
            if step < 8 * batch_num:
                return 1e-3
            else:
                return 1e-4

    # model = m.get_model()
    # save_dir = 'ckpt-vgg16'

    filters_size = np.array([1, 2, 4, 8, 16, 32], dtype=np.int)
    times = 8
    filters_size *= times
    model = m.get_small_model(filters_size, l2=l2)

    if cb_flag:
        init_nums = np.array([293, 51], dtype=np.int)
        trn_init_nums = (0.8 * init_nums).astype(np.int)
        aug_nums = 80 * trn_init_nums

        if init_cb_n:
            loss_type = 'cb_init_n'
            # cb_factor_list = cb.get_factor_list(sample_nums=aug_nums, unique_prototypes=trn_init_nums)
            cb_factor_list = 1.0 / aug_nums
        else:
            loss_type = 'cb_common_beta'
            cb_factor_list = cb.get_factor_list(sample_nums=aug_nums)
    else:
        cb_factor_list = None
        loss_type = 'ce'

    save_dir_prefix = 'ckpt'
    batch_size_str = 'b_' + str(batch_size)
    model_str = 'cnn_' + '_'.join([str(i) for i in filters_size])
    save_dir = '-'.join([save_dir_prefix, loss_type, batch_size_str, model_str])

    start_epoch = tf.Variable(initial_value=0, dtype=tf.int32)
    tp = tf.Variable(initial_value=0.0)
    fp = tf.Variable(initial_value=0.0)
    tn = tf.Variable(initial_value=0.0)
    fn = tf.Variable(initial_value=0.0)
    recall = tf.Variable(initial_value=0.0)
    # error_rate = tf.Variable(initial_value=0.0)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=DecayLr())
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.99)

    # checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, start_epoch=start_epoch,
    #                                  tp=tp, fp=fp, tn=tn, fn=fn, recall=recall, error_rate=error_rate)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, start_epoch=start_epoch,
                                     tp=tp, fp=fp, tn=tn, fn=fn, recall=recall)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if training:

        batch_num_str = 'batch num: %d' % batch_num
        # log(batch_num_str)
        val_batch_num = int(sample_loader.val_indices.shape[0] / batch_size)

        manager = tf.train.CheckpointManager(checkpoint=checkpoint, directory=save_dir, max_to_keep=1)

        summary_writer = tf.summary.create_file_writer('./tensorboard')

        max_recall = 0.0

        if retrain_flag:
            log('Load the checkpoint')
            checkpoint.restore(tf.train.latest_checkpoint(save_dir))
            log('Loaded')
            max_recall = recall.numpy()

        # log('Start training')
        for e in range(start_epoch.numpy(), epoch):

            sample_loader.shuffle_indices()
            for b in range(batch_num):
                start = b * batch_size
                end = (b + 1) * batch_size
                x, y = sample_loader.get_batch(start, end)
                with tf.GradientTape() as tape:
                    y_pred = model(x)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
                    if cb_flag:
                        batch_cb_factors = cb_factor_list[y]
                        batch_cb_factors = tf.constant(batch_cb_factors, dtype=tf.float32)
                        loss = tf.multiply(batch_cb_factors, loss)
                    loss = tf.reduce_mean(loss)
                    with summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=e * batch_num + b)
                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

            TP = tf.keras.metrics.TruePositives()
            FP = tf.keras.metrics.FalsePositives()
            TN = tf.keras.metrics.TrueNegatives()
            FN = tf.keras.metrics.FalseNegatives()

            for b in range(val_batch_num):
                start = b * batch_size
                end = (b + 1) * batch_size
                x, y = sample_loader.get_batch(start, end, val=True)
                y_pred = model(x)
                y_pred = tf.argmax(y_pred, axis=-1)
                TP.update_state(y, y_pred)
                FP.update_state(y, y_pred)
                TN.update_state(y, y_pred)
                FN.update_state(y, y_pred)

            tp.assign(TP.result())
            fp.assign(FP.result())
            tn.assign(TN.result())
            fn.assign(FN.result())
            recall.assign(tp / (tp + fn))
            # error_rate.assign(((fn / (tp + fn)) + (fn / (fp + tn))) / 2)

            # res = 'epoch:%d tp=%d fp=%d tn=%d fn=%d recall=%f error_rate=%f' % (
            #     e, tp.numpy(), fp.numpy(), tn.numpy(), fn.numpy(), recall.numpy(), error_rate.numpy())
            #
            # log(res)
            start_epoch.assign_add(1)
            if recall.numpy() > max_recall:
                manager.save(checkpoint_number=e)
                max_recall = recall.numpy()

    checkpoint.restore(tf.train.latest_checkpoint(save_dir))
    res = 'best val result: tp=%d fp=%d tn=%d fn=%d recall=%f' % (
        tp.numpy(), fp.numpy(), tn.numpy(), fn.numpy(), recall.numpy())

    # log(res)

    # log('Testing')
    TP = tf.keras.metrics.TruePositives()
    FP = tf.keras.metrics.FalsePositives()
    TN = tf.keras.metrics.TrueNegatives()
    FN = tf.keras.metrics.FalseNegatives()

    tst_num = sample_loader.x_tst.shape[0]  # 2880
    tst_batch_num = 10
    tst_batch_size = int(tst_num / 10)

    for i in range(tst_batch_num):
        start = i * tst_batch_size
        end = (i + 1) * tst_batch_size
        x = sample_loader.x_tst[start:end]
        # x = np.expand_dims(x, axis=0)
        y = sample_loader.y_tst[start:end]
        # y = np.expand_dims(y, axis=0)
        y_pred = model(x)
        y_pred = tf.argmax(y_pred, axis=-1)
        TP.update_state(y, y_pred)
        FP.update_state(y, y_pred)
        TN.update_state(y, y_pred)
        FN.update_state(y, y_pred)

    tp.assign(TP.result())
    fp.assign(FP.result())
    tn.assign(TN.result())
    fn.assign(FN.result())
    recall.assign(tp / (tp + fn))

    res = 'test result: tp=%d fp=%d tn=%d fn=%d recall=%f' % (
        tp.numpy(), fp.numpy(), tn.numpy(), fn.numpy(), recall.numpy())

    log(res)


if __name__ == '__main__':
    for i in range(5):
        # log(str(i))
        main()
