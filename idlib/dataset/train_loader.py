import numpy as np
import cv2
import os
import random
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class TrainLoader:
    def __init__(self, data_dir='data/'):
        self.data_dir = data_dir

    def _parse_labels(self):
        labels_file = os.path.join(self.data_dir, "ascii/forms.txt")
        data_list = list()
        with open(labels_file) as f:
            for i in range(16):
                next(f)
            for line in f:
                data_list.append(line.split())
        return data_list

    def _list_data(self, data_list):
        self.writers_forms_dict = dict()
        for data_point in data_list:
            if data_point[1] in self.writers_forms_dict.keys():
                if data_point[0][0] < 'e':
                    self.writers_forms_dict[data_point[1]].append(os.path.join(self.data_dir, f'formsA-D/{data_point[0]}.png'))
                elif data_point[0][0] < 'i':
                    self.writers_forms_dict[data_point[1]].append(os.path.join(self.data_dir, f'formsE-H/{data_point[0]}.png'))
                else:
                    self.writers_forms_dict[data_point[1]].append(os.path.join(self.data_dir, f'formsI-Z/{data_point[0]}.png'))
            else:
                if data_point[0][0] < 'e':
                    self.writers_forms_dict[data_point[1]] = [os.path.join(self.data_dir, f'formsA-D/{data_point[0]}.png')]
                elif data_point[0][0] < 'i':
                    self.writers_forms_dict[data_point[1]] = [os.path.join(self.data_dir, f'formsE-H/{data_point[0]}.png')]
                else:
                    self.writers_forms_dict[data_point[1]] = [os.path.join(self.data_dir, f'formsI-Z/{data_point[0]}.png')]

    def build_dataset(self):
        data_list = self._parse_labels()
        self._list_data(data_list)

    def get_data_samples(self, num_writers=3, num_train_samples=2):
        writers_list = [writer for writer in self.writers_forms_dict.keys() if len(self.writers_forms_dict[writer]) > num_train_samples]
        train_writers = random.choices(writers_list, k=num_writers)
        test_writer = random.choice(train_writers)
        label_encoder = LabelEncoder()
        label_encoder.fit(train_writers)
        xtrain = list()
        ytrain = list()
        xvalid = list()
        yvalid =  list()
        for writer in train_writers:
            if writer == test_writer:
                rand_choices = random.choices(self.writers_forms_dict[writer], k=num_train_samples+1)
                xvalid.append(random.choice(rand_choices))
                yvalid.append(writer)
                rand_choices.remove(xvalid)
                xtrain.extend(rand_choices)
                ytrain.extend([writer]*num_train_samples)
            else:
                xtrain.extend(random.choices(self.writers_forms_dict[writer], k=num_train_samples))
                ytrain.extend([writer]*num_train_samples)
        ytrain = label_encoder.transform(ytrain)
        yvalid = label_encoder.transform(yvalid)
        return xtrain, xvalid, ytrain, yvalid

    def get_complete_data(self, specific_writers=None, test_split=0.2):
        if not specific_writers:
            specific_writers = self.writers_forms_dict.keys()
        X = list()
        Y = list()
        for writer in specific_writers:
            X.extend(self.writers_forms_dict[writer])
            Y.extend([writer]*len(self.writers_forms_dict[writer]))
        label_encoder = LabelEncoder()
        label_encoder.fit(specific_writers)
        Y = label_encoder.transform(Y)
        xtrain, xvalid, ytrain, yvalid = train_test_split(X, Y, stratify=Y, random_state=42,
                                                            test_size=test_split, shuffle=True)
        return xtrain, xvalid, ytrain, yvalid
