# to use the new computer to write code
from math import log

def create_dataset():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels

test_ds, test_labels = create_dataset()

# write the code to calc Shannon entropy
def calc_shannon_ent(dataset):
    num_ent = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        curr_label = feat_vec[-1]
        if curr_label not in label_counts.keys():
            label_counts[curr_label] = 0
        label_counts[curr_label] += 1
    
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_ent
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent

calc_shannon_ent(test_ds)