"""Shared configuration for Phase 5 feature engineering."""

from __future__ import annotations

# Skewness / log transform
ABS_SKEWNESS_THRESHOLD = 1.0
MIN_UNIQUE_FOR_LOG = 10
MIN_NON_NULL_FOR_TRANSFORM = 20

# Binning
NUM_BINS = 5
MIN_UNIQUE_FOR_BINNING = 10

# Validation
GENERATED_FEATURE_MAX_MISSING_PERCENT = 50.0

# Reproducibility
RANDOM_STATE = 42

PREVIEW_ROWS = 10
