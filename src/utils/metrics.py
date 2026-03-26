import tensorflow as tf
import tf_keras as keras

class BinaryIoU(keras.metrics.Metric):
    def __init__(self, threshold=0.5, from_logits=False, name="binary_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.from_logits = from_logits
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")
        self.tn = self.add_weight(name="tn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert logits -> probabilities if the caller said so explicitly,
        # otherwise use the values as probabilities directly.
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true > 0.5, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)
        self.tn.assign_add(tn)

    def result(self):
        # Calculate Intersection over Union for foreground and background, returning the mean (mIoU)
        iou_fg = self.tp / (self.tp + self.fp + self.fn + 1e-7)
        iou_bg = self.tn / (self.tn + self.fp + self.fn + 1e-7)
        return (iou_fg + iou_bg) / 2.0

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)
        self.tn.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold, "from_logits": self.from_logits})
        return config
