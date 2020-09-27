from numpy import *

def create_dataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

group, labels = create_dataset()

def hu_classify(origin_point=[0, 0], group=group, labels=labels, k=3):
    distances = sum((tile(origin_point, reps=(group.shape[0], 1)) - group) ** 2, axis=1)
    distance_indices = distances.argsort()

    class_dis_pairs = {}
    for i in range(k):
        label_i = labels[distance_indices[i]]
        class_dis_pairs[label_i] = class_dis_pairs.get(label_i, 0) + 1
    # sort the dic by value in dict
    class_dis_pairs = sorted(class_dis_pairs.items(), key = lambda x: x[1], reverse=True)
    print(class_dis_pairs)
    # output the class of the point
    return class_dis_pairs[0][0]

# the function to read file
def parse_file(file_name) :
    fr = open(file_name)
    array_lines = fr.readlines()
    number_lines = len(array_lines)
    return_mat = zeros((number_lines, 3))

    class_label_vector = []
    index = 0

    for line in array_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    
    return return_mat, class_label_vector