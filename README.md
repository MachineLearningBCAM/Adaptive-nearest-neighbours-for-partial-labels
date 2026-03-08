# Adaptive-nearest-neighbours-for-partial-labels
The proposed files implement the Adaptive nearest neighbours methods for partial labels (PL AKNN) presented in .
## Code

All MATLAB source files are located in the `code/` folder.

### Algorithm
- **`PL_Aknn_classifier.m`** — Main algorithm. Implements the adaptive partial-label
  k-nearest neighbor classifier (PL-A-kNN). 

### Preprocessing
- **`nn_preprocess_vision.m`** — Preprocessing pipeline for the vision benchmarks
  (CIFAR-10, MNIST, Fashion-MNIST). 

- **`nn_preprocess_real_partial_datasets.m`** — Preprocessing pipeline for the
  real-world partial label datasets (MIRFlickr, MSRCv2). 
### Partial Label Generation
- **`candidate_generator_vision.m`** — Generates partial label scenarios for the
  vision benchmarks by introducing controlled ambiguity and noise into the label sets using
  a cluster-based candidate generation strategy.

- **`real_partial_datasets_with_noise.m`** — Adds synthetic noise to the existing
  partial label annotations of the real-world datasets (MIRFlickr, MSRCv2),
  simulating varying levels of label ambiguity.

### Utilities
- **`dataset_split_part_labels.m`** — Splits a dataset into training and test sets
  while preserving the partial label structure.
