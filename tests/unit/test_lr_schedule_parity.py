"""Stock callbacks must schedule exactly as the removed code did.

`train.update_plateau_and_early_stopping` was a hand-written
ReduceLROnPlateau + EarlyStopping, needed only because distillation ran its own
epoch loop. `DistillModel` moved that loop into `fit`, so the stock callbacks
took over. The semantics are close but not trivially identical, so this replays
one val_loss sequence through both and asserts they agree epoch by epoch.

`_removed_scheduler` below is a frozen copy of the deleted function's behaviour.
It exists only as the reference for this comparison; it is not called by
anything else and must not grow features.
"""

import tf_keras as keras

# A sequence with the two things worth testing: an improvement that resets both
# counters, then a long plateau that triggers two LR drops and finally the stop.
VAL_LOSSES = [1.0, 0.9, 0.9, 0.95, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87]

CONFIG = {
    "lr_plateau_patience": 2,
    "lr_plateau_factor": 0.5,
    "min_lr": 1e-4,
    "patience": 4,
}
INITIAL_LR = 1e-2


def _removed_scheduler(val_losses, config, initial_lr):
    """The behaviour of the deleted `update_plateau_and_early_stopping`."""
    lr = initial_lr
    best = float("inf")
    patience = lr_patience = 0
    trace = []

    for epoch, current in enumerate(val_losses):
        if current < best:
            best = current
            patience = lr_patience = 0
        else:
            patience += 1
            lr_patience += 1

        if lr_patience >= config["lr_plateau_patience"]:
            new_lr = max(lr * config["lr_plateau_factor"], config["min_lr"])
            if new_lr < lr:
                lr = new_lr
            lr_patience = 0

        stopped = patience >= config["patience"]
        trace.append((epoch, lr, stopped))
        if stopped:
            break
    return trace


def _stock_callbacks(val_losses, config, initial_lr):
    """The same sequence driven through Keras's own callbacks."""
    model = keras.Sequential([keras.layers.Dense(1, input_shape=(1,))])
    model.compile(optimizer=keras.optimizers.Adam(initial_lr), loss="mse")

    # cooldown=0 and min_delta=0 are spelled out because they are exactly what
    # makes Keras reset on any improvement, the way the removed code did.
    plateau = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config["lr_plateau_factor"],
        patience=config["lr_plateau_patience"],
        min_lr=config["min_lr"],
        min_delta=0.0,
        cooldown=0,
        verbose=0,
    )
    early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config["patience"],
        min_delta=0.0,
        restore_best_weights=True,
        verbose=0,
    )
    for callback in (plateau, early):
        callback.set_model(model)
        callback.on_train_begin()

    trace = []
    for epoch, current in enumerate(val_losses):
        logs = {"val_loss": current}
        plateau.on_epoch_end(epoch, logs)
        early.on_epoch_end(epoch, logs)
        lr = float(keras.backend.get_value(model.optimizer.learning_rate))
        stopped = bool(model.stop_training)
        trace.append((epoch, lr, stopped))
        if stopped:
            break
    return trace


def test_stock_callbacks_match_the_removed_scheduler():
    """Same learning rate at every epoch, and the same epoch to stop on."""
    removed = _removed_scheduler(VAL_LOSSES, CONFIG, INITIAL_LR)
    stock = _stock_callbacks(VAL_LOSSES, CONFIG, INITIAL_LR)

    assert len(removed) == len(stock), "the two schedulers stopped at different epochs"
    for (epoch, old_lr, old_stop), (_, new_lr, new_stop) in zip(removed, stock, strict=True):
        assert abs(old_lr - new_lr) < 1e-9, f"epoch {epoch}: lr {old_lr} vs {new_lr}"
        assert old_stop == new_stop, f"epoch {epoch}: stop {old_stop} vs {new_stop}"


def test_the_sequence_actually_exercises_both_mechanisms():
    """Guard against the parity test passing because nothing ever happened."""
    trace = _stock_callbacks(VAL_LOSSES, CONFIG, INITIAL_LR)
    learning_rates = [lr for _, lr, _ in trace]
    assert len(set(learning_rates)) >= 3, "the LR never dropped twice"
    assert trace[-1][2], "early stopping never triggered"


def test_learning_rate_is_clamped_at_min_lr():
    """The plateau schedule must never take the LR below min_lr."""
    config = dict(CONFIG, min_lr=5e-3)
    trace = _stock_callbacks(VAL_LOSSES, config, INITIAL_LR)
    # The learning rate is held as float32, so compare with float32 slack.
    assert min(lr for _, lr, _ in trace) >= config["min_lr"] * (1 - 1e-6)
