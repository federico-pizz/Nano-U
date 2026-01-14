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
                        tf.print("Warning: Regex pattern", sel, "matched no layers")
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
        return tf.keras.Model(inputs=self.model.inputs, outputs=outputs)

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
