# Writer Identification System

## Considered Research Papers
-   [Text independent writer recognition using redundant writing patterns with contour-based orientation and curvature features](https://drive.google.com/file/d/1bI3k3wCjC1TNK3C6hgXMy2W9TWHMXZff/view?usp=sharing).
-   [An improved online writer identification framework using codebook descriptors](https://drive.google.com/file/d/1VheUDrH_9d2-vJLz7tzTHsSrlb_EshG2/view?usp=sharing).
-   [Writer identification using texture features: A comparative study](https://drive.google.com/file/d/1MLogDf_XSJc4LUEn3ZI1WO1wnQVvl7jM/view?usp=sharing).
-   [Center Symmetric Local Binary Co-occurrence Pattern for Texture, Face and Bio-medical Image Retrieval](https://www.researchgate.net/publication/281559563_Center_Symmetric_Local_Binary_Co-occurrence_Pattern_for_Texture_Face_and_Bio-medical_Image_Retrieval)

## Dataset Exploration
-   The used dataset is [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database).
-   Refer to `experiments/dataset-exploration.ipynb` for dataset exploration.

## Pre-Processing Stage
-   [x] Remove document noise.
-   [x] Extract written part only.
-   [x] Segment out lines.

## Feature Extraction
-   [x] LBP Texture Descriptors.
-   [ ] GLCM (CSLBCoP) Texture Descriptors _[PENDING REVIEW]_.
-   [ ] LPQ Texture Descriptors _[PENDING REVIEW]_.
-   [X] PCA (or Truncated SVD) on extracted features.

## Classifiers
-   [x] SVM.
-   [x] KNN.
-   [x] RF.
-   [x] LR.
-   [x] NB.

## Installation

-   Install dependencies from `requirements.txt`:
    ```bash
    pip3 install -r requirements.txt
    ```

## Usage

-   Run `run.py` :
    ```bash
    python run.py -dir /path/to/data/root/directory -mode [complete-train | sampled-train | test]
    ```
