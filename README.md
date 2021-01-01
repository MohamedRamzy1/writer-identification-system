# Writer Identification System

## Considered Research Papers
-   [Text independent writer recognition using redundant writing patterns with contour-based orientation and curvature features](https://drive.google.com/file/d/1bI3k3wCjC1TNK3C6hgXMy2W9TWHMXZff/view?usp=sharing).
-   [An improved online writer identification framework using codebook descriptors](https://drive.google.com/file/d/1VheUDrH_9d2-vJLz7tzTHsSrlb_EshG2/view?usp=sharing).
-   [Writer identification using texture features: A comparative study](https://drive.google.com/file/d/1MLogDf_XSJc4LUEn3ZI1WO1wnQVvl7jM/view?usp=sharing).

## Dataset Exploration
-   The used dataset is [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).
-   Refer to `experiments/dataset-exploration.ipynb` for dataset exploration.

## Pre-Processing Stage
-   Remove document noise.
-   Extract written part only.
-   Segment out lines.

## Feature Extraction
-   Texture Descriptors (LBP mostly).
-   Codebooks _[OPTIONAL]_.

## Classifiers
-   SVM.
-   MLP.
-   K-NN.
