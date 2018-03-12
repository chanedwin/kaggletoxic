from utils import load_data, DATA_FILE

header, data = load_data(DATA_FILE, type='lst')
import numpy as np

positive_retained_list = []
negative_retained_list = []

for sentence in data:
    for point in sentence[2:7]:
        if point != "0":
            positive_retained_list.append(sentence)
            break

for sentence in data:
    all_zeroes = True
    for point in sentence[2:7]:
        if point != "0":
            all_zeroes = False
            break
    if all_zeroes:
        if np.random.rand() < 0.25:
            negative_retained_list.append(sentence)

all_list = positive_retained_list + negative_retained_list
all_list.sort(key=lambda x: x[0])

import csv

with open('./data/balanced_train_file.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(header)
    for line in all_list:
        spamwriter.writerow(line)
