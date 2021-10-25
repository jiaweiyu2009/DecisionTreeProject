import json
import math


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
        if(class1 == 0):
            p0 = class0/total
            p2 = class2/total
            entropy -= p0*math.log(p0, 2)
            entropy -= p2*math.log(p2, 2)
        if(class2 == 0):
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


def best_split_for_continuous(dataset):
    best_attribute = 0
    best_thresh_val = 0
    best_infoGain = 0
    old_info = getEntropy(counts_in_each_class(dataset))
    numberOfAttributes = 0
    # for x in range(numberOfAttributes):
    pointsToSplitOn = possibleThresholdValues(dataset, 0)
    for y in pointsToSplitOn:
        partition1, partition2 = splitInto2PartitionForContinuous(
            dataset, 0, y)


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


def splitInto2PartitionForContinuous(dataset, attribute, thresholdValue):
    partition1 = []
    partition2 = []

    for x in dataset:
        if(x[attribute] < thresholdValue):
            partition1.append(x)
        else:
            partition2.append(x)
    return partition1, partition2


train_data = json.load(open('test.json'))
modified_train_data = []
# dev_data = json.load(open('dev.json'))
for x in train_data['data']:
    modified_train_data.append(x)

for x in range(len(train_data['data'])):
    modified_train_data[x].append(train_data['label'][x])

print best_split_for_continuous(modified_train_data)
