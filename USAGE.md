# Test Usage

## Installation

-   Install dependencies from `requirements.txt`:
    ```bash
    pip3 install -r requirements.txt
    ```

## Usage

-   Run `run.py` :
    ```bash
    python run.py -dir /path/to/test/data/root/directory -mode test
    ```

`/path/to/test/data/root/directory` should contain `data` folder containing test cases. After running the previous command, the code generates two text files `results.txt` and `time.txt` in the root directory next to `data` folder.  The root folder should be of the following structure :

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

__NOTE :__ The test cases folders should be numbered in ascending order and padded with zeros to the same length, in order to keep the test cases ordered.
