from keras import backend as K


# def precision_micro(y_true, y_pred):
#     temp = y_true * (y_pred == 1)
#     temp = K.mean(temp, axis=0)
#     tp = K.sum(temp)
#     fp = K.sum((1 - temp))
#     return tp / (tp + fp)
#
#
# def recall_micro(y_true, y_pred):
#     temp = y_pred * (y_true == 1)
#     temp = K.mean(temp, axis=0)
#     tp = K.sum(temp)
#     fn = K.sum(1 - temp)
#     return tp / (tp + fn)


def precision_micro(y_true, y_pred):
    tp = K.sum(y_true * y_pred)
    fp = K.sum(y_pred * K.cast(K.equal(y_true, K.zeros_like(y_true)), "float32"))
    return tp / (tp + fp + K.epsilon())


def recall_micro(y_true, y_pred):
    tp = K.sum(y_true * y_pred)
    fn = K.sum(y_true * K.cast(K.equal(y_pred, K.zeros_like(y_pred)), "float32"))
    return tp / (tp + fn + K.epsilon())


def f1_micro(y_true, y_pred):
    y_pred = K.argmax(y_pred, axis=-1)
    y_pred = K.one_hot(y_pred, 12)
    pm = precision_micro(y_true, y_pred)
    rm = recall_micro(y_true, y_pred)
    return (2 * pm * rm) / (pm + rm + K.epsilon())
