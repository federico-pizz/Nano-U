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

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert logits → probabilities if the caller said so explicitly,
        # otherwise use the values as probabilities directly.
        # The previous auto-detection via per-batch min/max was unreliable:
        # early-training logits can fall inside [-0.1, 1.1] and be silently
        # treated as probabilities, giving garbage IoU values.
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true > 0.5, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        # Micro-averaged Jaccard index across all accumulated batches
        denom = self.tp + self.fp + self.fn + 1e-7
        return self.tp / denom

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"threshold": self.threshold, "from_logits": self.from_logits})
        return config
