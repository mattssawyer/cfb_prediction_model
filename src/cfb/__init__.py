"""CFB predictor package.

Public entry points:
- ``cfb.data.CFBDataLoader``: unified CFBD API loader with on-disk parquet cache.
- ``cfb.features.build_training_matrix``: rebuild the training-time feature matrix.
- ``cfb.features.build_week_features``: build the per-week feature matrix for inference.
- ``cfb.model.train``: end-to-end training + calibration.
- ``cfb.model.load``: load the booster + schema.
- ``cfb.model.predict_week``: high-level weekly-prediction workflow.
"""

__version__ = "0.1.0"
