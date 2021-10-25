import json
import math


def possibleThresholdValues(dataset, attribute):
    dlist = []
    pointsToSplitOn = []
    for x in range(len(dataset)):
        dlist.append(dataset[x][attribute])

    dlist.sort()
    pointsToSplitOn = []
    thresholdValues = []

    lowest = dlist[0] - 0.5
    highest = dlist[len(dlist)-1] + 0.5
    pointsToSplitOn.append(lowest)
    pointsToSplitOn.append(highest)
    for x in range(len(dlist) - 1):
        pointsToSplitOn.append((dlist[x]+dlist[x+1])/2)

    pointsToSplitOn.sort()
    return pointsToSplitOn


def best_split(dataset):
    old_info = getEntropy(counts_in_each_class(dataset))
    numberOfAttributes = len(dataset[0]) - 1
    first_attribute_tuple = best_values_for_concrete(dataset)
    best_attribute = 0
    best_thresh_val = first_attribute_tuple[1]
    best_infoGain = first_attribute_tuple[2]
    for x in range(1, numberOfAttributes):
        pointsToSplitOn = possibleThresholdValues(dataset, x)
        for y in pointsToSplitOn:
            partition1, partition2 = splitInto2PartitionForContinuous(
                dataset, x, y)

            if((len(partition1) == 0) or (len(partition2) == 0)):
                continue

            entropy1 = getEntropy(counts_in_each_class(partition1))
            entropy2 = getEntropy(counts_in_each_class(partition2))
            info_gain = old_info - informationForContinuous(
                len(partition1), len(partition2), entropy1, entropy2)

            if(info_gain >= best_infoGain):
                best_infoGain = info_gain
                best_attribute = x
                best_thresh_val = y

    return best_attribute, best_thresh_val, best_infoGain


def best_values_for_concrete(dataset):
    best_cat = 0
    best_infoGain = 0
    old_info = getEntropy(counts_in_each_class(dataset))
    pointsToSplitOn = [0, 1, 2]
    for y in pointsToSplitOn:
        partition1, partition2 = splitInto2PartitionForConcrete(
            dataset, 0, y)

        if((len(partition1) == 0) or (len(partition2) == 0)):
            continue

        entropy1 = getEntropy(counts_in_each_class(partition1))
        entropy2 = getEntropy(counts_in_each_class(partition2))
        info_gain = old_info - informationForContinuous(
            len(partition1), len(partition2), entropy1, entropy2)
        if(info_gain > best_infoGain):
            best_infoGain = info_gain
            best_cat = y

    return 0, best_cat, best_infoGain


def counts_in_each_class(dataset):
    class0 = 0
    class1 = 0
    class2 = 0
    for x in dataset:
        if(x[-1] == 0):
            class0 += 1
        elif(x[-1] == 1):
            class1 += 1
        else:
            class2 += 1
    return class0, class1, class2


def dominate_class(classCountTuple):
    class0 = classCountTuple[0]
    class1 = classCountTuple[1]
    class2 = classCountTuple[2]
    if (max(class0, class1, class2) == class0):
        return 0
    elif (max(class0, class1, class2) == class1):
        return 1
    else:
        return 2


def splitInto2PartitionForContinuous(dataset, attribute, thresholdValue):
    partition1 = []
    partition2 = []

    for x in dataset:
        if(x[attribute] < thresholdValue):
            partition1.append(x)
        else:
            partition2.append(x)
    return partition1, partition2


def splitInto2PartitionForConcrete(dataset, attribute, category):
    partition1 = []
    partition2 = []

    for x in dataset:
        if(x[attribute] == category):
            partition1.append(x)
        else:
            partition2.append(x)
    return partition1, partition2


def getEntropy(classCountTuple):
    class0 = classCountTuple[0]
    class1 = classCountTuple[1]
    class2 = classCountTuple[2]
    total = class0 + class1 + class2
    entropy = 0
    if (class0 == 0 and class2 == 0) or (class0 == 0 and class1 == 0) or (class1 == 0 and class2 == 0):
        entropy = 0
    else:
        if(class0 == 0):
            p1 = class1/total
            p2 = class2/total
            entropy -= p1*math.log(p1, 2)
            entropy -= p2*math.log(p2, 2)
        elif(class1 == 0):
            p0 = class0/total
            p2 = class2/total
            entropy -= p0*math.log(p0, 2)
            entropy -= p2*math.log(p2, 2)
        elif(class2 == 0):
            p0 = class0/total
            p1 = class1/total
            entropy -= p0*math.log(p0, 2)
            entropy -= p1*math.log(p1, 2)
        else:
            p0 = class0/total
            p1 = class1/total
            p2 = class2/total
            entropy -= p0*math.log(p0, 2)
            entropy -= p1*math.log(p1, 2)
            entropy -= p2*math.log(p2, 2)
    return entropy


def informationForContinuous(partition1, partition2, entropy1, entropy2):
    total = partition1 + partition2
    information = ((partition1/total)*entropy1) + ((partition2/total)*entropy2)
    return information


def informationForConcrete(partition1, partition2, partition3, entropy1, entropy2, entropy3):
    total = partition1 + partition2 + partition3
    information = ((partition1/total)*entropy1) + \
        ((partition2/total)*entropy2) + ((partition3/total)*entropy3)
    return information


class leave_node:
    def __init__(self, dataset):
        self.label = dominate_class(counts_in_each_class(dataset))


class not_leave_node:
    def __init__(self, attribute, attribute_value, partition1, partition2):
        self.attribute = attribute
        self.attribute_value = attribute_value
        self.partition1 = partition1
        self.partition2 = partition2


def grow_tree(dataset):
    best_attribute, best_attribute_value, best_infoGain = best_split(dataset)
    if(best_infoGain == 0):
        return leave_node(dataset)

    if(best_attribute == 0):
        partition1, partition2 = splitInto2PartitionForConcrete(
            dataset, best_attribute, best_attribute_value)
    else:
        partition1, partition2 = splitInto2PartitionForContinuous(
            dataset, best_attribute, best_attribute_value)

    new_partition1 = grow_tree(partition1)
    new_partition2 = grow_tree(partition2)

    return not_leave_node(best_attribute, best_attribute_value, new_partition1, new_partition2)


def match_data_with_split_attribute(data, attribute, attribute_value):
    if(attribute == 0):
        if(data[0] == attribute_value):
            return True
        else:
            return False
    else:
        if(data[attribute] < attribute_value):
            return True
        else:
            return False


def predict_data(data, tree_node):
    if isinstance(tree_node, leave_node):
        return tree_node.label
    else:
        if(match_data_with_split_attribute(data, tree_node.attribute, tree_node.attribute_value) == True):
            return predict_data(data, tree_node.partition1)
        else:
            return predict_data(data, tree_node.partition2)


correct = 0
train_data = json.load(open('train.json'))
test_data = json.load(open('dev.json'))
modified_train_data = []
# dev_data = json.load(open('dev.json'))
for x in train_data['data']:
    modified_train_data.append(x)

for x in range(len(train_data['data'])):
    modified_train_data[x].append(train_data['label'][x])

trained_tree = grow_tree(modified_train_data)

modified_test_data = []
for x in test_data['data']:
    modified_test_data.append(x)

for x in range(len(test_data['data'])):
    modified_test_data[x].append(test_data['label'][x])

predict_label_list = [0] * len(modified_test_data)
for x in range(len(modified_test_data)):
    predict_label_list[x] = predict_data(modified_test_data[x], trained_tree)

for x in range(len(predict_label_list)):
    if (predict_label_list[x] == modified_test_data[x][-1]):
        correct += 1
print(correct)
print(len(predict_label_list))
