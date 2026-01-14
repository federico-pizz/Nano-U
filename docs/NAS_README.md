# NAS utilities in this project

This document explains the NAS-related utilities added to the project and how to use them in the training pipeline.

Overview
- src/nas_covariance.py contains utilities to measure and regularize channel-level redundancy inside activation layers. The goal is to help Neural Architecture Search (NAS) or pruning decisions by quantifying how redundant channels are within a layer.

Key components
- ActivationExtractor
  - Build a cached Keras Model that returns intermediate layer activations for a set of selected layer names or regex selectors.
  - Use when you need to extract activations for regularization or distillation without rebuilding models repeatedly.

- _collapse_spatial_to_channels(x, data_format)
  - Collapse spatial dims into a [B, C] matrix by averaging over spatial dimensions. Supports rank 2-5 tensors and both channels_last and channels_first formats.

- covariance_redundancy(activations, normalize=True, return_metrics=False)
  - Computes a DeCov-style redundancy score. Higher means more redundancy.
  - Optional return_metrics flag returns diagnostics such as trace, off_diagonal_norm and condition number.

- FeatureDecorrelationRegularizer
  - Keras regularizer wrapper exposing the redundancy penalty for use directly in layer regularizers.
  - Supports warmup (linear scaling) to avoid penalizing early training.

- covariance_regularizer_loss
  - Aggregates redundancy penalties across multiple activations (e.g., selected conv layers) with optional weighting and per-layer reporting.

- RunningCovariance
  - TF-native Welford running covariance estimator useful for streaming or validation-set analysis.
  - Exposes redundancy_score() computed from accumulated covariance.

- DistillationAwareNAS
  - Small helper to compute feature-matching losses between teacher and student models and analyze layer redundancy for NAS decisions.

Usage patterns
- Per-step regularization
  - Create an ActivationExtractor for the set of layer names you want to regularize.
  - During training, call extractor(inputs, training=True) to obtain activations and pass them to covariance_regularizer_loss.
  - Add the returned scalar to your task loss with an appropriate multiplier.

- Offline analysis for NAS/pruning
  - Use DistillationAwareNAS.analyze_layer_importance with a validation dataset to get per-layer redundancy scores.
  - Lower scores indicate less redundancy (more useful channels), higher scores indicate candidates for pruning.

Testing
- Unit tests for core utilities are in tests/test_nas_covariance.py. Running tests requires a TF-enabled virtual environment.

Notes and integration
- Warnings are emitted with tf.print to avoid noisy stdout. If you prefer the project logger, replace tf.print calls with your logging facility.
- The module removed inline examples to avoid accidental running. See the plans/nas_covariance_changelog.md for details.

Contact
- If you need these utilities adapted to a different training loop or to support per-sample covariances, open an issue or assign to the maintainer.
