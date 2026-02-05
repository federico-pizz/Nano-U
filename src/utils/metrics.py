import tensorflow as tf

class BinaryIoU(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="binary_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Auto-detect logits: if values are outside [0.0, 1.0], apply sigmoid
        # Use graph-compatible operations
        max_val = tf.reduce_max(y_pred)
        min_val = tf.reduce_min(y_pred)
        is_logit = tf.logical_or(min_val < -0.1, max_val > 1.1)
        
        y_pred = tf.where(is_logit, tf.nn.sigmoid(y_pred), y_pred)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true > 0.5, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        denom = self.tp + self.fp + self.fn + 1e-7
        return self.tp / denom

    def reset_states(self):
        for v in self.variables:
            v.assign(0)
