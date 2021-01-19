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
-   [x] GLCM Texture Descriptors.
-   [x] GLCM (CSLBCoP) Texture Descriptors.
-   [x] LPQ Texture Descriptors.
-   [X] PCA (or Truncated SVD) on extracted features.

## Classifiers
-   [x] SVM.
-   [x] KNN.
-   [x] RF.
-   [x] LR.
-   [x] NB.

## Folder Structure

-   `idlib/classifier` : contains code for different classifiers.
-   `idlib/dataset` : contains code for train and test dataloaders.
-   `idlib/feature_extractor` : contains code for different feature extractors.
-   `idlib/preprocessor` : contains code for form preparation.
-   `idlib/trainer` : contains code for training functions.
-   `idlib/test.py` : contains main _test_ pipeline.
-   `idlib/train.py` : contains main _train_ pipeline.
-   `run.py` : contains main driver function for system.

## Installation

-   Install dependencies from `requirements.txt`:
    ```bash
    pip3 install -r requirements.txt
    ```

## Usage

### Complete Train

-   Run `run.py` :
    ```bash
    python run.py -dir /path/to/data/root/directory -mode complete-train
    ```

`/path/to/data/root/directory` should contain `formsA-D`, `formsE-H`, `formsI-Z` and `ascii` folders from `IAM Handwriting Database`.

### Sampled Train

-   Run `run.py` :
    ```bash
    python run.py -dir /path/to/data/root/directory -mode sampled-train
    ```

`/path/to/data/root/directory` should contain `formsA-D`, `formsE-H`, `formsI-Z` and `ascii` folders from `IAM Handwriting Database`.

### Test

-   Run `run.py` :
    ```bash
    python run.py -dir /path/to/test/data/root/directory -mode test
    ```

`/path/to/test/data/root/directory` should contain `data` folder containing test cases and should be of the following structure :

```
.
└── data
    ├── 01
    │   ├── 1
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 2
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 3
    │   │   ├── 1.png
    │   │   └── 2.png
    │   └── test.png
    ├── 02
    │   ├── 1
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 2
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 3
    │   │   ├── 1.png
    │   │   └── 2.png
    │   └── test.png
    ├── 03
    │   ├── 1
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 2
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 3
    │   │   ├── 1.png
    │   │   └── 2.png
    │   └── test.png
    ├── 04
    │   ├── 1
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 2
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 3
    │   │   ├── 1.png
    │   │   └── 2.png
    │   └── test.png
    ├── 05
    │   ├── 1
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 2
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 3
    │   │   ├── 1.png
    │   │   └── 2.png
    │   └── test.png
    ├── 06
    │   ├── 1
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 2
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 3
    │   │   ├── 1.png
    │   │   └── 2.png
    │   └── test.png
    ├── 07
    │   ├── 1
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 2
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 3
    │   │   ├── 1.png
    │   │   └── 2.png
    │   └── test.png
    ├── 08
    │   ├── 1
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 2
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 3
    │   │   ├── 1.png
    │   │   └── 2.png
    │   └── test.png
    ├── 09
    │   ├── 1
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 2
    │   │   ├── 1.png
    │   │   └── 2.png
    │   ├── 3
    │   │   ├── 1.png
    │   │   └── 2.png
    │   └── test.png
    └── 10
        ├── 1
        │   ├── 1.png
        │   └── 2.png
        ├── 2
        │   ├── 1.png
        │   └── 2.png
        ├── 3
        │   ├── 1.png
        │   └── 2.png
        └── test.png
```

After running the previous command, the code generates two text files `results.txt` and `time.txt` in the root directory next to `data` folder.
