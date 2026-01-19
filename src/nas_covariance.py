"""
Utilities for measuring neuron covariance inside activation layers and
regularizers/hooks for NAS integration.

This module provides:
- ActivationExtractor: cached extractor for intermediate activations
- Utilities to collapse spatial dims into channel activations
- covariance_redundancy: DeCov-style redundancy score
- FeatureDecorrelationRegularizer: Keras regularizer that wraps redundancy
- RunningCovariance: Welford-style online covariance estimator (TF-safe)
- DistillationAwareNAS: small helper to compare teacher/student activations
- NASMonitorCallback: Keras callback for non-invasive NAS monitoring

The implementation removes examples and keeps only runtime utilities used by
the project pipeline. Non-obvious implementation details are documented inline
in the functions/classes below.
"""

from typing import Dict, List, Optional, Union, Tuple
import tensorflow as tf
import re


class ActivationExtractor:
    """
    Cached extractor for intermediate activations.

    Purpose
    - Build a single intermediate Keras model that returns the outputs of
      selected layers. Rebuilding this model repeatedly is expensive; the
      extractor caches it.

    Selection semantics
    - Each selector can be either:
      * exact layer name (string not wrapped in slashes)
      * regex pattern string wrapped in slashes, e.g. /conv_\d+/
      * a Layer instance (will use layer.name)

    Notes
    - Resolved layer names are deduplicated while preserving order.
    """

    def __init__(self, model: tf.keras.Model, layer_selectors: List[Union[str, tf.keras.layers.Layer]]):
        self.model = model
        self.layer_selectors = layer_selectors
        self.layer_names = self._resolve_layer_names()
        self.intermediate_model = self._build_intermediate_model()

    def _resolve_layer_names(self) -> List[str]:
        resolved: List[str] = []
        all_layers = self.model.layers
        layer_names_set = {l.name for l in all_layers}

        for sel in self.layer_selectors:
            if isinstance(sel, str):
                # Regex selection: strings wrapped with /pattern/
                if len(sel) >= 2 and sel.startswith('/') and sel.endswith('/'):
                    pattern = re.compile(sel[1:-1])
                    matches = [l.name for l in all_layers if pattern.search(l.name)]
                    # Use tf.print to avoid noisy stdout in some environments
                    if not matches:
                        tf.print("Warning: Regex pattern", sel, "matched no layers. Attempting fallback search by substring 'conv'.")
                        # Fallback: look for common conv substring (case-insensitive)
                        fallback = [l.name for l in all_layers if 'conv' in l.name.lower()]
                        if fallback:
                            tf.print("Fallback matched layers:", fallback)
                            matches = fallback
                    resolved.extend(matches)
                else:
                    # exact name
                    if sel not in layer_names_set:
                        raise ValueError(f"Layer '{sel}' not found in model. Available layers: {list(layer_names_set)[:10]}...")
                    resolved.append(sel)
            else:
                # assume layer instance
                if sel.name not in layer_names_set:
                    raise ValueError(f"Layer instance '{sel.name}' not found in model")
                resolved.append(sel.name)

        # deduplicate while preserving order
        seen = set()
        out = []
        for n in resolved:
            if n not in seen:
                seen.add(n)
                out.append(n)
        if not out:
            # No selectors matched; fail fast with helpful diagnostics
            available = sorted(list(layer_names_set))
            raise ValueError(f"No layers matched selectors {self.layer_selectors}. Available layer names (first 20): {available[:20]}")
        return out

    def _build_intermediate_model(self) -> tf.keras.Model:
        outputs = []
        for name in self.layer_names:
            try:
                layer = self.model.get_layer(name)
                outputs.append(layer.output)
            except ValueError:
                raise ValueError(f"Layer '{name}' not found in model.")
        # Single model that maps original inputs to a list of intermediate outputs
        if not outputs:
            raise ValueError(f"No intermediate outputs resolved for selectors: {self.layer_selectors}")
        return tf.keras.Model(inputs=self.model.inputs, outputs=outputs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Ensure cached intermediate model is released when used as a context manager
        self.cleanup()
        return False

    def __call__(self, inputs: tf.Tensor, training: bool = False) -> Dict[str, tf.Tensor]:
        outs = self.intermediate_model(inputs, training=training)
        # Ensure we always return a dict of name -> tensor
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        return {name: out for name, out in zip(self.layer_names, outs)}

    def cleanup(self):
        """Release cached intermediate model to free memory.

        Call when the extractor will no longer be used (for long-running
        processes or to avoid holding model references that block cleanup).
        """
        if hasattr(self, 'intermediate_model'):
            del self.intermediate_model


class UNetLayerSelector:
    """Convenience helpers to select U-Net related layers by name patterns."""

    @staticmethod
    def get_encoder_layers(model: tf.keras.Model,
                          pattern: str = r'(down|encoder|contract).*conv') -> List[str]:
        pattern_re = re.compile(pattern, re.IGNORECASE)
        return [l.name for l in model.layers if pattern_re.search(l.name)]

    @staticmethod
    def get_decoder_layers(model: tf.keras.Model,
                          pattern: str = r'(up|decoder|expand).*conv') -> List[str]:
        pattern_re = re.compile(pattern, re.IGNORECASE)
        return [l.name for l in model.layers if pattern_re.search(l.name)]

    @staticmethod
    def get_bottleneck_layers(model: tf.keras.Model,
                              pattern: str = r'(bottleneck|bridge)') -> List[str]:
        pattern_re = re.compile(pattern, re.IGNORECASE)
        return [l.name for l in model.layers if pattern_re.search(l.name)]

    @staticmethod
    def get_skip_connection_layers(model: tf.keras.Model,
                                   pattern: str = r'(skip|concat|add)') -> List[str]:
        pattern_re = re.compile(pattern, re.IGNORECASE)
        return [l.name for l in model.layers if pattern_re.search(l.name)]

    @staticmethod
    def get_all_conv_layers(model: tf.keras.Model) -> List[str]:
        conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.Conv3D,
                     tf.keras.layers.SeparableConv2D, tf.keras.layers.DepthwiseConv2D)
        return [l.name for l in model.layers if isinstance(l, conv_types)]


def _collapse_spatial_to_channels(x: tf.Tensor,
                                  data_format: str = 'channels_last') -> tf.Tensor:
    """
    Collapse spatial dimensions of a tensor into a [batch, channels] matrix.

    Accepts rank 2,3,4,5 tensors and handles channels_first / channels_last.

    Behavior details:
    - rank==2: assumed already [B, C], returned unchanged
    - rank==3: collapsed one spatial dimension -> reduce_mean over that dim
    - rank==4: typical NHWC or NCHW image tensor -> reduce_mean over H and W
    - rank==5: 3D convs (D, H, W) -> reduce_mean over spatial dims
    - other ranks: fallback to reshaping to [B, -1]

    The function prefers static shape checks when available to avoid control
    flow overhead; it falls back to a tf.switch_case when rank is dynamic.
    """
    if x is None:
        raise ValueError("activations tensor is None")

    rank_static = x.shape.rank
    if rank_static is not None:
        if rank_static == 2:
            return x
        if rank_static == 4:
            axes = [1, 2] if data_format == 'channels_last' else [2, 3]
            # mean over spatial dims -> [B, C]
            return tf.reduce_mean(x, axis=axes)
        if rank_static == 3:
            axis = 1 if data_format == 'channels_last' else 2
            return tf.reduce_mean(x, axis=axis)
        if rank_static == 5:  # 3D convolutions
            axes = [1, 2, 3] if data_format == 'channels_last' else [2, 3, 4]
            return tf.reduce_mean(x, axis=axes)
        # fallback: reshape any other rank into [B, -1]
        shape = tf.shape(x)
        return tf.reshape(x, [shape[0], -1])

    # dynamic-rank branch: use tf.switch_case to avoid Python branching in graph
    rank = tf.rank(x)

    def _case2():
        return x

    def _case4():
        axes = [1, 2] if data_format == 'channels_last' else [2, 3]
        return tf.reduce_mean(x, axis=axes)

    def _case3():
        axis = 1 if data_format == 'channels_last' else 2
        return tf.reduce_mean(x, axis=axis)

    def _case5():
        axes = [1, 2, 3] if data_format == 'channels_last' else [2, 3, 4]
        return tf.reduce_mean(x, axis=axes)

    def _default():
        shape = tf.shape(x)
        return tf.reshape(x, [shape[0], -1])

    return tf.switch_case(rank, branch_fns={2: _case2, 3: _case3, 4: _case4, 5: _case5},
                         default=_default)


def covariance_redundancy(activations: tf.Tensor,
                          normalize: bool = True,
                          eps: float = 1e-8,
                          data_format: str = 'channels_last',
                          return_metrics: bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, Dict]]:
    """
    Compute a DeCov-style redundancy score for a batch of activations.

    Implementation notes
    - Activations are first collapsed to [B, C] using _collapse_spatial_to_channels.
    - Data is centered and a sample covariance matrix is computed (using B-1 denom).
    - Off-diagonal Frobenius norm (squared) is used as the redundancy measure.
    - Optional normalization divides by trace to make the score scale-invariant.
    - Small noise is added to centered activations and tiny entries clamped for
      numerical stability when computing eigenvalues / condition numbers.

    Returns
    - scalar redundancy score (higher = more redundancy). If return_metrics True,
      returns a tuple (score, metrics_dict) with additional diagnostics.
    """
    x = _collapse_spatial_to_channels(activations, data_format)
    x = tf.cast(x, tf.float32)
    shape = tf.shape(x)
    B = tf.cast(shape[0], tf.float32)
    C = tf.cast(shape[1], tf.float32)

    def compute():
        # Center the data across the batch
        x_centered = x - tf.reduce_mean(x, axis=0, keepdims=True)

        # Small isotropic noise helps stabilise covariance computations when some
        # channels are nearly constant; the scale is eps * 0.1 which is tiny.
        x_centered = x_centered + tf.random.normal(tf.shape(x_centered), 0.0, eps * 0.1)

        denom = tf.maximum(B - 1.0, 1.0)
        cov_mat = tf.matmul(x_centered, x_centered, transpose_a=True) / denom

        # Clamp extremely small values to zero to avoid spurious eigenvalues
        cov_mat = tf.where(tf.abs(cov_mat) < eps, 0.0, cov_mat)

        diag = tf.linalg.diag_part(cov_mat)
        full_sq = tf.reduce_sum(tf.square(cov_mat))
        diag_sq = tf.reduce_sum(tf.square(diag))
        off_sq = tf.maximum(full_sq - diag_sq, 0.0)
        off_loss = 0.5 * off_sq

        if normalize:
            trace = tf.reduce_sum(diag)
            result = off_loss / tf.maximum(trace, eps)
        else:
            result = off_loss / tf.maximum(C * (C - 1.0), 1.0)

        if return_metrics:
            # eigenvalues used for condition number; small eps prevents div by zero
            eigenvalues = tf.linalg.eigvalsh(cov_mat)
            condition_number = tf.reduce_max(eigenvalues) / (tf.reduce_min(eigenvalues) + eps)

            metrics = {
                'redundancy_score': result,
                'trace': trace if normalize else tf.reduce_sum(diag),
                'off_diagonal_norm': tf.sqrt(off_sq),
                'condition_number': condition_number,
                'mean_correlation': off_sq / tf.maximum(C * (C - 1.0) * trace, eps)
            }
            return result, metrics

        return result

    result = tf.cond(
        tf.logical_or(tf.less_equal(B, 1.0), tf.less_equal(C, 1.0)),
        lambda: (tf.constant(0.0, dtype=tf.float32), {}) if return_metrics else tf.constant(0.0, dtype=tf.float32),
        compute
    )

    return result


class FeatureDecorrelationRegularizer(tf.keras.regularizers.Regularizer):
    """
    Keras regularizer that applies the DeCov redundancy term to activation tensors.

    warmup_steps (optional)
    - If provided, the regularizer scales up linearly over warmup_steps calls so
      that early training is not penalized heavily.
    """

    def __init__(self,
                 l2: float = 0.01,
                 normalize: bool = True,
                 data_format: str = 'channels_last',
                 warmup_steps: Optional[int] = None):
        self.l2 = l2
        self.normalize = normalize
        self.data_format = data_format
        self.warmup_steps = warmup_steps
        self._step = tf.Variable(0, dtype=tf.int32, trainable=False) if warmup_steps else None

    def __call__(self, x):
        # Compute redundancy score on activations
        score = covariance_redundancy(x, normalize=self.normalize, data_format=self.data_format)
        score = tf.cast(score, tf.float32)

        # Warmup scheduling: increment internal counter and scale factor
        if self.warmup_steps is not None:
            self._step.assign_add(1)
            warmup_factor = tf.minimum(
                tf.cast(self._step, tf.float32) / tf.cast(self.warmup_steps, tf.float32),
                1.0
            )
            return tf.multiply(tf.cast(self.l2, tf.float32) * warmup_factor, score)

        return tf.multiply(tf.cast(self.l2, tf.float32), score)

    def get_config(self):
        config = {
            'l2': float(self.l2),
            'normalize': self.normalize,
            'data_format': self.data_format
        }
        if self.warmup_steps is not None:
            config['warmup_steps'] = int(self.warmup_steps)
        return config


def covariance_regularizer_loss(activations: Dict[str, tf.Tensor],
                                essential_layers: Optional[List[str]] = None,
                                weights: Optional[Dict[str, float]] = None,
                                normalize: bool = True,
                                reduction: str = 'mean',
                                data_format: str = 'channels_last',
                                return_layer_scores: bool = False) -> Union[tf.Tensor, Tuple[tf.Tensor, Dict]]:
    """
    Aggregate redundancy losses for multiple activation tensors.

    Args
    - activations: mapping layer_name -> activation tensor
    - essential_layers: if provided, only these layers are considered
    - weights: optional per-layer weights
    - reduction: 'mean' or 'sum'
    - return_layer_scores: if True, also return per-layer raw scores
    """
    losses = []
    layer_scores = {}
    targets = essential_layers or list(activations.keys())

    for name in targets:
        if name not in activations:
            tf.print(f"Warning: Layer {name} not found in activations")
            continue

        score = covariance_redundancy(activations[name], normalize, data_format=data_format)
        w = weights.get(name, 1.0) if weights else 1.0
        weighted_score = score * w
        losses.append(weighted_score)

        if return_layer_scores:
            layer_scores[name] = score

    if not losses:
        total_loss = tf.constant(0.0, dtype=tf.float32)
        return (total_loss, layer_scores) if return_layer_scores else total_loss

    stacked = tf.stack(losses)
    if reduction == 'mean':
        total_loss = tf.reduce_mean(stacked)
    elif reduction == 'sum':
        total_loss = tf.reduce_sum(stacked)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    return (total_loss, layer_scores) if return_layer_scores else total_loss


class RunningCovariance:
    """
    Welford-style online covariance estimator implemented with TF variables.

    Notes on implementation
    - Uses tf.Variable for n (count), mean, and M2 (sum of outer products)
    - update() is decorated with tf.function and uses tf.cond to ensure the
      update logic is executed as TF ops in graph mode (no Python-side state).
    - _update_logic is intentionally written to return None and perform in-place
      Variable updates via assign/assign_add so it composes correctly inside
      tf.cond.
    """

    def __init__(self, channels: int, data_format: str = 'channels_last'):
        self.channels = channels
        self.data_format = data_format
        self.n = tf.Variable(0, dtype=tf.int32, trainable=False)
        # Use float64 internally for better numeric stability over many updates
        self.mean = tf.Variable(tf.zeros(channels, dtype=tf.float64), trainable=False)
        self.M2 = tf.Variable(tf.zeros((channels, channels), dtype=tf.float64), trainable=False)

    @tf.function
    def update(self, x: tf.Tensor):
        """
        Incorporate a batch of activations into running statistics.

        Behavior:
        - If the incoming batch has zero examples (B == 0) the update is a no-op.
        - Otherwise, _update_logic performs the Welford merge of batch stats into
          the running variables. Using tf.cond ensures TF graph execution.
        """
        x = _collapse_spatial_to_channels(x, self.data_format)
        x = tf.cast(x, tf.float64)
        B = tf.shape(x)[0]

        # tf.cond requires both branches to be callable TF functions; we use a
        # no-op lambda for the empty-batch case and the _update_logic path for
        # the non-empty case.
        tf.cond(
            tf.equal(B, 0),
            lambda: None,
            lambda: self._update_logic(x, B)
        )

    def _update_logic(self, x: tf.Tensor, B: tf.Tensor):
        """
        Merge batch statistics into running mean and M2.

        Steps:
        1. compute batch mean and centered batch samples
        2. compute batch M2 (unnormalized covariance sum) as centered^T * centered
        3. if this is the first update (self.n == 0), adopt batch stats
        4. otherwise, update self.M2 and self.mean using the pairwise merge formula
        """
        batch_mean = tf.reduce_mean(x, axis=0)
        delta = batch_mean - self.mean
        tot_n = self.n + B

        centered = x - batch_mean
        # batch_M2 is unnormalized sum of outer products across the batch
        batch_M2 = tf.matmul(centered, centered, transpose_a=True)

        def _first_update():
            # straightforward assign when no previous data exists
            self.M2.assign(batch_M2)
            self.mean.assign(batch_mean)
            self.n.assign(B)
            return None

        def _accumulate_update():
            # delta_outer accounts for the between-batch contribution
            delta_outer = tf.einsum('i,j->ij', delta, delta) * tf.cast(self.n * B / tot_n, tf.float64)
            self.M2.assign_add(batch_M2 + delta_outer)
            # update mean in-place
            self.mean.assign_add(delta * tf.cast(B / tot_n, tf.float64))
            self.n.assign(tot_n)
            return None

        tf.cond(tf.equal(self.n, 0), _first_update, _accumulate_update)

    def covariance(self) -> tf.Tensor:
        """Return the sample covariance matrix (shape [C, C])."""
        return tf.cond(
            self.n <= 1,
            lambda: tf.zeros((self.channels, self.channels), dtype=tf.float64),
            lambda: self.M2 / tf.cast(self.n - 1, tf.float64)
        )

    def redundancy_score(self, normalize: bool = True, eps: float = 1e-8) -> tf.Tensor:
        """Compute redundancy score from accumulated covariance statistics."""
        cov = self.covariance()
        diag = tf.linalg.diag_part(cov)
        full_sq = tf.reduce_sum(tf.square(cov))
        diag_sq = tf.reduce_sum(tf.square(diag))
        off_sq = tf.maximum(full_sq - diag_sq, 0.0)
        off_loss = 0.5 * off_sq

        if normalize:
            trace = tf.reduce_sum(diag)
            return off_loss / tf.maximum(trace, eps)
        C = tf.cast(self.channels, tf.float64)
        return off_loss / tf.maximum(C * (C - 1.0), 1.0)

    def reset(self):
        """Reset running statistics to zero state."""
        self.n.assign(0)
        self.mean.assign(tf.zeros(self.channels, dtype=tf.float64))
        self.M2.assign(tf.zeros((self.channels, self.channels), dtype=tf.float64))


class DistillationAwareNAS:
    """
    Small helper to compute feature-matching distillation losses between
    teacher and student for a mapping of corresponding layers.

    This class relies on ActivationExtractor to fetch layer activations from
    both models and exposes two utilities used by NAS pipelines: compute_distillation_loss
    and analyze_layer_importance.
    """

    def __init__(self,
                 teacher_model: tf.keras.Model,
                 student_model: tf.keras.Model,
                 layer_mapping: Dict[str, str]):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.layer_mapping = layer_mapping

        # Extractors map teacher layer names (keys) and student layer names (values)
        self.teacher_extractor = ActivationExtractor(teacher_model, list(layer_mapping.keys()))
        self.student_extractor = ActivationExtractor(student_model, list(layer_mapping.values()))

    def compute_distillation_loss(self,
                                  inputs: tf.Tensor,
                                  temperature: float = 3.0,
                                  alpha: float = 0.5) -> tf.Tensor:
        """Return mean MSE across mapped feature pairs.

        Note: temperature and alpha are placeholders for combined soft-targets +
        feature matching schemes; currently only simple feature MSE is implemented.
        """
        teacher_acts = self.teacher_extractor(inputs, training=False)
        student_acts = self.student_extractor(inputs, training=True)

        feature_losses = []
        for t_name, s_name in self.layer_mapping.items():
            t_act = teacher_acts[t_name]
            s_act = student_acts[s_name]

            # simple mean-squared error between features
            loss = tf.reduce_mean(tf.square(t_act - s_act))
            feature_losses.append(loss)

        return tf.reduce_mean(feature_losses)

    def analyze_layer_importance(self,
                                 validation_data: tf.data.Dataset,
                                 num_batches: int = 50) -> Dict[str, float]:
        """Return redundancy scores per teacher layer using RunningCovariance.

        For each selected teacher layer, accumulate running covariance statistics
        over num_batches batches from validation_data and return the computed
        redundancy score as a float.
        """
        layer_redundancies = {name: RunningCovariance(
            # infer channel count from layer output shape; last dim expected
            int(self.teacher_model.get_layer(name).output.shape[-1])
        ) for name in self.layer_mapping.keys()}

        for i, (x, _) in enumerate(validation_data.take(num_batches)):
            acts = self.teacher_extractor(x, training=False)
            for name, act in acts.items():
                layer_redundancies[name].update(act)

        scores = {}
        for name, rc in layer_redundancies.items():
            scores[name] = float(rc.redundancy_score().numpy())

        return scores

    def cleanup(self):
        self.teacher_extractor.cleanup()
        self.student_extractor.cleanup()


class NASMonitorCallback(tf.keras.callbacks.Callback):
    """
    Non-invasive NAS monitoring via Keras callback.
    
    Monitors feature redundancy during training by computing covariance
    statistics on model outputs OR internal layer activations. Works with any
    model architecture.
    
    Args:
        validation_data: Validation dataset (tf.data.Dataset or tuple of arrays).
                        Required if you want to monitor redundancy during training.
                        Pass the same validation_data you use in model.fit().
        layer_selectors: Optional list of layer names or regex patterns to monitor.
                        If None, monitors model output (may return zeros for single-channel output).
                        Examples: ['encoder_conv1', 'bottleneck'], or ['/conv.*/'] for regex.
        monitor_frequency: 'batch' or 'epoch' - when to compute metrics
        log_frequency: Log every N batches/epochs
        redundancy_weight: Optional weight for adding redundancy to loss (not implemented yet)
        log_dir: Directory for TensorBoard logs
        save_history: Save metrics to CSV at end of training
        csv_path: Path for CSV file (default: 'nas_metrics.csv')
    
    Example (monitor output):
        >>> val_ds = ...  # your validation dataset
        >>> callback = NASMonitorCallback(
        ...     validation_data=val_ds,
        ...     monitor_frequency='epoch',
        ...     log_frequency=1,
        ...     log_dir='logs/nas',
        ...     save_history=True
        ... )
        >>> model.fit(train_ds, validation_data=val_ds, callbacks=[callback])
    
    Example (monitor internal layers):
        >>> callback = NASMonitorCallback(
        ...     validation_data=val_ds,
        ...     layer_selectors=['encoder_conv_0', 'encoder_conv_1', 'bottleneck'],
        ...     monitor_frequency='epoch',
        ...     log_dir='logs/nas'
        ... )
        >>> model.fit(train_ds, validation_data=val_ds, callbacks=[callback])
    """
    
    def __init__(self,
                 validation_data=None,
                 layer_selectors=None,
                 monitor_frequency='epoch',
                 log_frequency=1,
                 redundancy_weight=0.0,
                 log_dir=None,
                 save_history=True,
                 csv_path='nas_metrics.csv'):
        super().__init__()
        self.validation_data = validation_data
        self.layer_selectors = layer_selectors
        self.monitor_frequency = monitor_frequency
        self.log_frequency = log_frequency
        self.redundancy_weight = redundancy_weight
        self.log_dir = log_dir
        self.save_history = save_history
        self.csv_path = csv_path
        
        # Extractor will be initialized in on_train_begin when model is available
        self.extractor = None
        self._extractor_initialized = False
        
        # Metrics storage
        self.redundancy_history = []
        self.correlation_history = []
        self.trace_history = []
        self.condition_history = []
        self.step_history = []
        self.batch_count = 0
        
        # TensorBoard writer
        if log_dir:
            self.tb_writer = tf.summary.create_file_writer(log_dir)
        else:
            self.tb_writer = None
    
    def on_train_begin(self, logs=None):
        """Initialize ActivationExtractor when training starts and model is available."""
        if self.layer_selectors and not self._extractor_initialized:
            try:
                # Check if model is a nested functional wrapper around subclassed model
                inner_model = None
                for layer in self.model.layers:
                    if isinstance(layer, tf.keras.Model) and layer.name in ['BU_Net', 'Nano_U']:
                        inner_model = layer
                        print(f"✓ NAS: Detected nested subclassed model: {layer.name}")
                        break
                
                if inner_model is None:
                    # Not a nested model - try standard approach
                    self.extractor = ActivationExtractor(self.model, self.layer_selectors)
                    self._extractor_initialized = True
                    print(f"✓ NAS: ActivationExtractor initialized with layers: {self.extractor.layer_names}")
                    return
                
                # For subclassed models, we need to create a new functional model
                # that exposes the intermediate layer outputs
                print(f"✓ NAS: Building functional extraction model for subclassed {inner_model.name}...")
                
                # Build the inner model if not already built
                if not inner_model.built:
                    input_shape = self.model.input_shape
                    if isinstance(input_shape, tuple) and input_shape[0] is None:
                        dummy_shape = (1,) + input_shape[1:]
                    else:
                        dummy_shape = input_shape
                    dummy_input = tf.zeros(dummy_shape)
                    _ = inner_model(dummy_input, training=False)
                
                # Collect target layer instances from model attributes
                target_layers = []
                for selector in self.layer_selectors:
                    found = False
                    # Check encoder convs
                    if hasattr(inner_model, 'enc_convs'):
                        for conv_layer in inner_model.enc_convs:
                            if conv_layer.name == selector:
                                target_layers.append(conv_layer)
                                found = True
                                break
                    # Check bottleneck
                    if not found and hasattr(inner_model, 'bottleneck'):
                        if inner_model.bottleneck.name == selector:
                            target_layers.append(inner_model.bottleneck)
                            found = True
                    # Check decoder convs
                    if not found and hasattr(inner_model, 'dec_convs'):
                        for conv_layer in inner_model.dec_convs:
                            if conv_layer.name == selector:
                                target_layers.append(conv_layer)
                                found = True
                                break
                    if not found:
                        print(f"⚠ Warning: Layer '{selector}' not found in model")
                
                if not target_layers:
                    raise ValueError(f"No valid layers found for selectors: {self.layer_selectors}")
                
                # Create a functional model that traces through the subclassed model
                # and captures the intermediate outputs
                print(f"✓ NAS: Creating functional extraction model with {len(target_layers)} target layers...")
                
                # Use the outer functional model's input
                inputs = self.model.input
                
                # Create a custom call function that extracts intermediate outputs
                @tf.function
                def extract_activations(x):
                    outputs = {}
                    # We need to trace through the model's call and capture outputs
                    # This is done by monkey-patching the layer call temporarily
                    for layer in target_layers:
                        original_call = layer.call
                        
                        def make_capturing_call(layer_name):
                            def capturing_call(inputs_inner, *args, **kwargs):
                                result = original_call(inputs_inner, *args, **kwargs)
                                outputs[layer_name] = result
                                return result
                            return capturing_call
                        
                        layer.call = make_capturing_call(layer.name)
                    
                    # Run forward pass
                    _ = inner_model(x, training=False)
                    
                    # Restore original calls
                    for layer in target_layers:
                        # Note: in practice this is tricky; better approach below
                        pass
                    
                    return outputs
                
                # Actually, let's use a simpler approach: create a custom extractor function
                self._target_layers = target_layers
                self._inner_model = inner_model
                self._extractor_initialized = True
                print(f"✓ NAS: Custom extractor initialized for layers: {[l.name for l in target_layers]}")
                    
            except Exception as e:
                import traceback
                print(f"⚠ Warning: Failed to initialize ActivationExtractor: {e}")
                print(f"   Traceback: {traceback.format_exc()}")
                print(f"   Available model layers: {[l.name for l in self.model.layers]}")
                print("   Falling back to output monitoring")
                self.extractor = None
                self._target_layers = None
                self._inner_model = None
                self._extractor_initialized = False
    
    def on_epoch_end(self, epoch, logs=None):
        """Monitor redundancy at end of each epoch."""
        if self.monitor_frequency != 'epoch':
            return
        
        if epoch % self.log_frequency != 0:
            return
        
        # Check if validation data is available
        if self.validation_data is None:
            # No validation data - skip metrics computation
            return
        
        # Compute metrics on first validation batch
        val_data = self.validation_data
        if isinstance(val_data, tf.data.Dataset):
            for x_val, y_val in val_data.take(1):
                metrics = self._compute_metrics(x_val)
                self._log_metrics(epoch, metrics, prefix='epoch')
                break
        elif isinstance(val_data, tuple) and len(val_data) >= 2:
            # Validation data passed as (x_val, y_val) tuple
            x_val = val_data[0]
            # Take a batch if it's a full dataset
            if len(x_val.shape) > 1:
                batch_size = min(32, len(x_val))
                x_batch = x_val[:batch_size]
                metrics = self._compute_metrics(x_batch)
                self._log_metrics(epoch, metrics, prefix='epoch')
    
    def on_batch_end(self, batch, logs=None):
        """Monitor redundancy at end of each batch (if enabled)."""
        if self.monitor_frequency != 'batch':
            return
        
        self.batch_count += 1
        if self.batch_count % self.log_frequency != 0:
            return
        
        # Batch-level monitoring requires access to training batch
        # This is more complex and optional for initial implementation
        pass
    
    def on_train_end(self, logs=None):
        """Save metrics to CSV at end of training."""
        if self.save_history and len(self.redundancy_history) > 0:
            self.save_metrics_csv(self.csv_path)
    
    def _compute_metrics(self, x):
        """Compute redundancy metrics on model output or internal layers."""
        if self.extractor is not None and self._extractor_initialized:
            # Monitor internal layers using ActivationExtractor
            try:
                activations = self.extractor(x, training=False)
                
                # Aggregate metrics across all monitored layers
                all_metrics = []
                for layer_name, act in activations.items():
                    redundancy_score, metrics_dict = covariance_redundancy(
                        act,
                        normalize=True,
                        return_metrics=True
                    )
                    all_metrics.append({
                        'layer': layer_name,
                        'redundancy_score': float(redundancy_score.numpy()),
                        'trace': float(metrics_dict.get('trace', 0)),
                        'mean_correlation': float(metrics_dict.get('mean_correlation', 0)),
                        'condition_number': float(metrics_dict.get('condition_number', 1))
                    })
                
                # Average metrics across layers
                if all_metrics:
                    return {
                        'redundancy_score': sum(m['redundancy_score'] for m in all_metrics) / len(all_metrics),
                        'trace': sum(m['trace'] for m in all_metrics) / len(all_metrics),
                        'mean_correlation': sum(m['mean_correlation'] for m in all_metrics) / len(all_metrics),
                        'condition_number': sum(m['condition_number'] for m in all_metrics) / len(all_metrics)
                    }
            except Exception as e:
                print(f"⚠ Warning: Failed to compute metrics from ActivationExtractor: {e}")
                print("   Falling back to output monitoring")
        
        elif hasattr(self, '_target_layers') and self._target_layers and self._extractor_initialized:
            # Custom extractor for subclassed models
            try:
                # Capture layer outputs by monkey-patching
                layer_outputs = {}
                original_calls = {}
                
                # Store original calls and replace with capturing wrappers
                for layer in self._target_layers:
                    original_calls[layer.name] = layer.call
                    
                    def make_capturing_wrapper(layer_name, original_call):
                        def wrapper(inputs, *args, **kwargs):
                            output = original_call(inputs, *args, **kwargs)
                            layer_outputs[layer_name] = output
                            return output
                        return wrapper
                    
                    layer.call = make_capturing_wrapper(layer.name, original_calls[layer.name])
                
                # Run forward pass through inner model
                _ = self._inner_model(x, training=False)
                
                # Restore original calls
                for layer in self._target_layers:
                    layer.call = original_calls[layer.name]
                
                # Compute metrics on captured activations
                all_metrics = []
                for layer_name, act in layer_outputs.items():
                    redundancy_score, metrics_dict = covariance_redundancy(
                        act,
                        normalize=True,
                        return_metrics=True
                    )
                    all_metrics.append({
                        'layer': layer_name,
                        'redundancy_score': float(redundancy_score.numpy()),
                        'trace': float(metrics_dict.get('trace', 0)),
                        'mean_correlation': float(metrics_dict.get('mean_correlation', 0)),
                        'condition_number': float(metrics_dict.get('condition_number', 1))
                    })
                
                # Average metrics across layers
                if all_metrics:
                    return {
                        'redundancy_score': sum(m['redundancy_score'] for m in all_metrics) / len(all_metrics),
                        'trace': sum(m['trace'] for m in all_metrics) / len(all_metrics),
                        'mean_correlation': sum(m['mean_correlation'] for m in all_metrics) / len(all_metrics),
                        'condition_number': sum(m['condition_number'] for m in all_metrics) / len(all_metrics)
                    }
            except Exception as e:
                import traceback
                print(f"⚠ Warning: Failed to compute metrics from custom extractor: {e}")
                print(f"   Traceback: {traceback.format_exc()}")
                print("   Falling back to output monitoring")
        
        # Fallback: Monitor model output
        y_pred = self.model(x, training=False)
        
        # Compute covariance-based redundancy with full metrics
        redundancy_score, metrics_dict = covariance_redundancy(
            y_pred,
            normalize=True,
            return_metrics=True
        )
        
        return {
            'redundancy_score': float(redundancy_score.numpy()),
            'trace': float(metrics_dict.get('trace', 0)),
            'mean_correlation': float(metrics_dict.get('mean_correlation', 0)),
            'condition_number': float(metrics_dict.get('condition_number', 1))
        }
    
    def _log_metrics(self, step, metrics, prefix='epoch'):
        """Log metrics to console, TensorBoard, and history."""
        # Console logging
        print(f"\nNAS Metrics ({prefix} {step}):")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
        
        # TensorBoard logging
        if self.tb_writer:
            with self.tb_writer.as_default():
                for key, value in metrics.items():
                    tf.summary.scalar(f'nas/{key}', value, step=step)
                self.tb_writer.flush()
        
        # History storage
        if self.save_history:
            self.redundancy_history.append(metrics['redundancy_score'])
            self.correlation_history.append(metrics.get('mean_correlation', 0))
            self.trace_history.append(metrics.get('trace', 0))
            self.condition_history.append(metrics.get('condition_number', 1))
            self.step_history.append(step)
    
    def get_metrics(self):
        """Return collected metrics as a dictionary."""
        return {
            'steps': self.step_history,
            'redundancy': self.redundancy_history,
            'correlation': self.correlation_history,
            'trace': self.trace_history,
            'condition_number': self.condition_history
        }
    
    def save_metrics_csv(self, filepath='nas_metrics.csv'):
        """Save metrics to CSV file."""
        import csv
        import os
        
        # Create directory if needed
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'redundancy_score', 'mean_correlation', 'trace', 'condition_number'])
            for i in range(len(self.step_history)):
                writer.writerow([
                    self.step_history[i],
                    self.redundancy_history[i],
                    self.correlation_history[i],
                    self.trace_history[i],
                    self.condition_history[i]
                ])
        
        print(f"✓ NAS metrics saved to {filepath}")
