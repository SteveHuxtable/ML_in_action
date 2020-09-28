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

def split_dataset(dataset, axis, value):
    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value: 
            reduce_feat_vec = feat_vec[:axis]
            reduce_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduce_feat_vec)
    return ret_dataset

split_dataset(test_ds, axis=0, value=1)

def choose_feat_to_split(dataset):
    num_features = len(dataset[0]) - 1
    base_entropy = calc_shannon_ent(dataset)
    best_info_gain = 0.0; best_feature = -1
    for i in range(num_features):
        feature_list = [example[i] for example in dataset]
        unique_vals = set(feature_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_shannon_ent(sub_dataset)
        info_gain = base_entropy - new_entropy
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature

choose_feat_to_split(test_ds)

test_list = ['a', 'a', 'c', 'a', 'c', 'a', 'c']

def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys(): class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
    return sorted_class_count[0][0]

majority_cnt(test_list)

def create_tree(dataset, labels):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    best_feat = choose_feat_to_split(dataset)
    best_feat_label = labels[best_feat]

    my_tree = {best_feat_label: {}}
    del (labels[best_feat])
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), sub_labels)
    return my_tree

create_tree(test_ds, test_labels)

del(test_list[2])