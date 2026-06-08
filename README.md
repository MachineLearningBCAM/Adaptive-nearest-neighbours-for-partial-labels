# Adaptive-nearest-neighbours-for-partial-labels
The proposed files implement the Adaptive Nearest Neighbors method for Partial Labels --PL A-kNN (https://arxiv.org/abs/2603.15781) presented in AISTATS 2026.
## Code

All MATLAB source files are located in the `code/` folder.

### Algorithm
- **`PL_Aknn_classifier.m`** — Main algorithm. Implements the Adaptive 
  Nearest Neighbor method for Partial Labels (PL A-kNN). In cases where two or more  labels remain in the candidate set
  after T iterations performs the heuristic desambiguatuon criterion specified in the paper.

### Preprocessing
- **`nn_preprocess_vision.m`** — Preprocessing pipeline for the vision benchmarks
  (CIFAR-10, MNIST, Fashion-MNIST). 

- **`nn_preprocess_real_partial_datasets.m`** — Preprocessing pipeline for the
  real-world partial label datasets (MIRFlickr, MSRCv2). 
### Partial Label Generation
- **`candidate_generator_vision.m`** — Generates partial labels scenarios for the
  vision benchmarks as specified in the paper.

- **`real_partial_datasets_with_noise.m`** — Adds synthetic noise to the existing
  partial label annotations of the real-world datasets (MIRFlickr, MSRCv2).

### Utilities
- **`dataset_split_part_labels.m`** — Splits a dataset into training and test sets
  while preserving the partial label structure.
## Quick Start

The `examples/` folder provides minimal runnable scripts that demonstrate 
how to use the method on any dataset and   setting specified in the paper. 


- **`vision_script.m`** — Example script for the vision benchmarks 
  (CIFAR-10, MNIST, Fashion-MNIST). Loads a dataset, generates a partial 
  label scenario with a configurable noise level, preprocesses the features, 
  and runs PL A-kNN returning the prediction accuracy.

- **`real_partial_datasets_script.m`** — Example script for the real-world partial 
  label datasets (MIRFlickr, MSRCv2). Loads a dataset, adds configurable 
  noise to the existing partial labels, preprocesses the features, and runs 
  PL A-kNN returning the prediction accuracy.

The key parameters that can be adjusted at the top of each script are:
- `noise` — (0.0 to 1.0)
- `T` — maximum number of iterations
- `c_1` —  hyperparameter of PL-A-kNN
- `prop_train` — train/test split proportion

## Author and Support

**Nicolas A. Errandonea**
nerrandonea@bcamath.org

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.

## Citation

If you find this code useful in your research, please include an explicit mention 
of our work in your publication with the following entry in your bibliography:


```bibtex
@inproceedings{errandonea2026learnability,
  title     = {Learnability with Partial Labels and Adaptive Nearest Neighbors},
  author    = {Errandonea, Nicol{\'a}s A. and Mazuelas, Santiago and Lozano, Jose A. and Dasgupta, Sanjoy},
  booktitle = {Proceedings of The 29th International Conference on Artificial Intelligence and Statistics},
  year      = {2026},
  url       = {https://openreview.net/forum?id=9Xf9ZUEF3F}
}
```
