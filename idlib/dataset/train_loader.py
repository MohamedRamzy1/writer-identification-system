import numpy as np
import os
import random
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class TrainLoader:
    """
    Train dataloader class
    ...

    Used to list the whole dataset or generate random samples
    of the dataset to match the test environment.
    ...

    Attributes
    ----------
    data_dir : str
        root directory of the dataset
    """
    def __init__(self, data_dir='data/'):
        # initialize parameters
        self.data_dir = data_dir

    def _parse_labels(self):
        # parse dataset labels text file (for forms only)
        # get text file directory
        labels_file = os.path.join(self.data_dir, "ascii/forms.txt")
        # read text label lines
        data_list = list()
        with open(labels_file) as f:
            # skip the first 16 lines (comments)
            for i in range(16):
                next(f)
            for line in f:
                data_list.append(line.split())
        return data_list

    def _list_data(self, data_list):
        # list the whole dataset into a single dictionary
        self.writers_forms_dict = dict()
        # loop over all form data points
        for data_point in data_list:
            # check whether the writer is already added
            if data_point[1] in self.writers_forms_dict.keys():
                # specify forms folder based on first letter
                if data_point[0][0] < 'e':
                    self.writers_forms_dict[data_point[1]].append(os.path.join(self.data_dir, f'formsA-D/{data_point[0]}.png'))
                elif data_point[0][0] < 'i':
                    self.writers_forms_dict[data_point[1]].append(os.path.join(self.data_dir, f'formsE-H/{data_point[0]}.png'))
                else:
                    self.writers_forms_dict[data_point[1]].append(os.path.join(self.data_dir, f'formsI-Z/{data_point[0]}.png'))
            else:
                # specify forms folder based on first letter
                if data_point[0][0] < 'e':
                    self.writers_forms_dict[data_point[1]] = [os.path.join(self.data_dir, f'formsA-D/{data_point[0]}.png')]
                elif data_point[0][0] < 'i':
                    self.writers_forms_dict[data_point[1]] = [os.path.join(self.data_dir, f'formsE-H/{data_point[0]}.png')]
                else:
                    self.writers_forms_dict[data_point[1]] = [os.path.join(self.data_dir, f'formsI-Z/{data_point[0]}.png')]

    def build_dataset(self):
        # build forms dataset dictionary
        data_list = self._parse_labels()
        self._list_data(data_list)

    def get_data_samples(self, num_writers=3, num_train_samples=2):
        # get a random dataset sample (consists of specific number of writers and samples per writer)
        # get all writers with number of forms more that samples per writer
        writers_list = [writer for writer in self.writers_forms_dict.keys() if len(self.writers_forms_dict[writer]) > num_train_samples]
        # select random writers
        train_writers = random.choices(writers_list, k=num_writers)
        # select a random test writer from train writers
        test_writer = random.choice(train_writers)
        # define and fit label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(train_writers)
        # list train and validation data samples
        xtrain = list()
        ytrain = list()
        xvalid = list()
        yvalid =  list()
        for writer in train_writers:
            # check whether the writer is selected as a test to select one more form
            if writer == test_writer:
                # select random forms from writer
                rand_choices = random.choices(self.writers_forms_dict[writer], k=num_train_samples+1)
                # get a single test form
                xvalid.append(random.choice(rand_choices))
                yvalid.append(writer)
                rand_choices.remove(xvalid)
                # append to xtrain and ytrain
                xtrain.extend(rand_choices)
                ytrain.extend([writer]*num_train_samples)
            else:
                # select random forms from writer and append to xtrain and ytrain
                xtrain.extend(random.choices(self.writers_forms_dict[writer], k=num_train_samples))
                ytrain.extend([writer]*num_train_samples)
        # encode labels
        ytrain = label_encoder.transform(ytrain)
        yvalid = label_encoder.transform(yvalid)
        return xtrain, xvalid, ytrain, yvalid

    def get_complete_data(self, forms_per_writer=5, test_split=0.2):
        # get complete dataset of writers with specific number of forms
        # list all forms for the specified writers
        X = list()
        Y = list()
        for writer in self.writers_forms_dict.keys():
            if len(self.writers_forms_dict[writer]) == forms_per_writer:
                X.extend(self.writers_forms_dict[writer])
                Y.extend([writer]*forms_per_writer)
        # encode labels
        label_encoder = LabelEncoder()
        label_encoder.fit(Y)
        Y = label_encoder.transform(Y)
        # train / validation split with stratify
        xtrain, xvalid, ytrain, yvalid = train_test_split(X, Y, stratify=Y, random_state=42,
                                                            test_size=test_split, shuffle=True)
        return xtrain, xvalid, ytrain, yvalid
