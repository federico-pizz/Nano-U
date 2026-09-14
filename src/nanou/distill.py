"""`DistillModel` - the knowledge-distillation training step, as a Keras model.

The supervised term, the distillation term and the Tversky blend are written
once here and used by both `train_step` and `test_step`, so the validation
objective cannot drift from the training objective. Before this class they were
two separate pieces of code, one eager and one graph, that had to be kept in
agreement by hand.

One of the two modules that touch Keras 2 internals directly (`qat.py` is the
other), so a later framework migration stays a contained job: Keras 3 changes
the custom-step contract (`compute_loss`, metric handling) and this file is
where that change would land.
"""

import tensorflow as tf
import tf_keras as keras

from nanou.losses import tversky_loss
from nanou.metrics import BinaryIoU

# How the distillation term is rescaled after temperature scaling.
#   "squared" - multiply by T^2, the classical Hinton correction
#   "none"    - leave the term at its natural magnitude
T_SCALES = ("squared", "none")


class DistillModel(keras.Model):
    """A student trained against ground truth and a frozen teacher at once.

    The objective is

        total = (1 - alpha) * supervised + alpha * distill
        supervised = (1 - tversky_weight) * BCE + tversky_weight * Tversky(a_fp, b_fn)
        distill = mean((sigmoid(t/T) - sigmoid(s/T))^2) * t_scale

    `alpha = 1.0` is a supported setting, not an edge case: it is how the CE-off
    ablation is expressed (see `search.make_config_overrides`). It zeroes the
    supervised term by multiplication, so there is no divide and no NaN.

    The tracked metric names are `loss`, `student_loss`, `distillation_loss` and
    `binary_iou`; `fit` prefixes the validation copies with `val_`. Those eight
    keys are what `pipeline._select_best_epoch_metrics` and every `results.json`
    consumer read, so they are part of the contract and must not be renamed.
    """

    def __init__(
        self,
        student: keras.Model,
        teacher: keras.Model,
        alpha: float = 0.3,
        temperature: float = 4.0,
        tversky_weight: float = 0.0,
        tversky_alpha: float = 0.7,
        tversky_beta: float = 0.3,
        t_scale: str = "squared",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if t_scale not in T_SCALES:
            raise ValueError(f"t_scale must be one of {T_SCALES}, got {t_scale!r}")

        self.student = student
        self.teacher = teacher
        # The teacher is a fixed target, never a thing being fitted. Freezing it
        # here is what keeps it out of `self.trainable_variables`, so the
        # optimizer cannot touch it however it is called.
        self.teacher.trainable = False

        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.tversky_weight = float(tversky_weight)
        self.tversky_alpha = float(tversky_alpha)
        self.tversky_beta = float(tversky_beta)
        self.t_scale = t_scale

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.student_loss_tracker = keras.metrics.Mean(name="student_loss")
        self.distill_loss_tracker = keras.metrics.Mean(name="distillation_loss")
        self.iou_metric = BinaryIoU(threshold=0.5, from_logits=True, name="binary_iou")

        self.bce_loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

    @property
    def metrics(self):
        """The four trackers, in history-key order. Keras resets these each epoch."""
        return [
            self.loss_tracker,
            self.student_loss_tracker,
            self.distill_loss_tracker,
            self.iou_metric,
        ]

    def call(self, inputs, training=False):
        """Forward the student. The teacher only ever runs inside the loss."""
        return self.student(inputs, training=training)

    def _supervised_loss(self, y_true, student_logits):
        """BCE, optionally blended with a precision-favoring Tversky term.

        `tversky_weight = 0.0` is pure BCE, which is what every shipped preset
        trains with; the blend exists so a student can be made deliberately
        conservative (alpha_fp > beta_fn penalises a false 'path' pixel harder
        than a missed one).
        """
        bce_term = self.bce_loss_fn(y_true, student_logits)
        if self.tversky_weight > 0.0:
            tv_term = tversky_loss(
                y_true,
                student_logits,
                alpha_fp=self.tversky_alpha,
                beta_fn=self.tversky_beta,
            )
            return (1.0 - self.tversky_weight) * bce_term + self.tversky_weight * tv_term
        return bce_term

    def _distillation_loss(self, teacher_logits, student_logits):
        """MSE between the temperature-scaled sigmoids of teacher and student."""
        temperature = self.temperature
        raw = tf.reduce_mean(
            tf.math.squared_difference(
                tf.nn.sigmoid(teacher_logits / temperature),
                tf.nn.sigmoid(student_logits / temperature),
            )
        )
        # Paper multiplies by T**2, the Hinton correction for the 1/T gradient
        # shrink that temperature scaling introduces. That reasoning assumes
        # T > 1. The tuned TinyAgri temperature is 0.7, which *sharpens* the
        # teacher instead of softening it, and T**2 = 0.49 then down-weights the
        # KD term rather than compensating for anything. Set
        # `training.t_scale: none` in a preset to drop the factor and separate
        # the two effects - that is the whole experiment.
        if self.t_scale == "squared":
            return raw * (temperature**2)
        return raw

    def _losses_for(self, x, y, training):
        """Compute (total, supervised, distillation, student logits) for one batch."""
        teacher_logits = self.teacher(x, training=False)
        student_logits = self.student(x, training=training)

        student_loss = self._supervised_loss(y, student_logits)
        distill_loss = self._distillation_loss(teacher_logits, student_logits)
        total_loss = (1.0 - self.alpha) * student_loss + self.alpha * distill_loss
        return total_loss, student_loss, distill_loss, student_logits

    def _update_trackers(self, y, total_loss, student_loss, distill_loss, student_logits):
        """Feed the four trackers and return the dict `fit` turns into history."""
        self.loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(student_loss)
        self.distill_loss_tracker.update_state(distill_loss)
        self.iou_metric.update_state(y, student_logits)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        """One gradient step on the student. The teacher is never updated."""
        x, y = data
        with tf.GradientTape() as tape:
            total_loss, student_loss, distill_loss, student_logits = self._losses_for(
                x, y, training=True
            )

        trainable = self.student.trainable_variables
        gradients = tape.gradient(total_loss, trainable)
        self.optimizer.apply_gradients(zip(gradients, trainable, strict=True))
        return self._update_trackers(y, total_loss, student_loss, distill_loss, student_logits)

    def test_step(self, data):
        """The training step minus the gradient - the same objective, by construction."""
        x, y = data
        total_loss, student_loss, distill_loss, student_logits = self._losses_for(
            x, y, training=False
        )
        return self._update_trackers(y, total_loss, student_loss, distill_loss, student_logits)
