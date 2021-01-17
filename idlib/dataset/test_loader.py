import os
import cv2


class TestLoader:
    """
    Test dataloader class
    ...

    Used to list the test cases using the provided test environment.
    ...

    Attributes
    ----------
    data_dir : str
        root directory of the test dataset
    num_writers_per_test : int
        number of writers per test cases
    num_samples_per_writer : int
        number of samples per writer
    """
    def __init__(
            self, data_dir='data/test_samples/',
            num_writers_per_test=3, num_samples_per_writer=2):
        # initialize parameters
        self.data_dir = os.path.join(data_dir, 'data')
        self.num_writers_per_test = num_writers_per_test
        self.num_samples_per_writer = num_samples_per_writer

    def list_test_cases(self):
        # list provided test cases
        return sorted(os.listdir(self.data_dir))

    def get_test_case(self, test_case_name):
        # get a specific test case
        # get test case directory
        case_dir = os.path.join(self.data_dir, test_case_name)
        # list data provided within test case
        xtrain_dir = list()
        ytrain = list()
        xtest_dir = os.path.join(case_dir, 'test.png')
        for writer in range(self.num_writers_per_test):
            writer_dir = os.path.join(case_dir, str(writer+1))
            writer_samples = [os.path.join(writer_dir, f'{str(sample+1)}.png')
                              for sample in range(self.num_samples_per_writer)]
            xtrain_dir.extend(writer_samples)
            ytrain.extend([writer]*self.num_samples_per_writer)
        # read form sample images
        xtrain = list()
        xtest = cv2.imread(xtest_dir, cv2.IMREAD_GRAYSCALE)
        for sample in xtrain_dir:
            xtrain.append(cv2.imread(sample, cv2.IMREAD_GRAYSCALE))
        # return train lists and test sample
        return xtrain, ytrain, xtest
