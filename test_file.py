import kNN_manual

group, labels = kNN_manual.create_dataset()

kNN_manual.hu_classify()
kNN_manual.hu_classify([2, 3], group=group, labels=labels, k=3)