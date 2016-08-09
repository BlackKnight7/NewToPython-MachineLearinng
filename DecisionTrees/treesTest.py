import trees

myDat, lables = trees.createDataSet()

print("------ shannon ------")
print(myDat)
print(trees.calcShannonEnt(myDat))

# print("------ shannon after changed ------")
# myDat[0][-1] = 'maybe'
# print(myDat)
# print(trees.calcShannonEnt(myDat))

print("------ split data set ------")
print(trees.splitDataSet(myDat, 0, 1))
print(trees.splitDataSet(myDat, 0, 0))

print("------ choose best feature to split ------")
print(trees.chooseBestFeatureToSplit(myDat))

print("------ create tree ------")
tree = trees.createTree(myDat, lables)
print(tree)

print("------ test tree classify ------")
print(trees.classify(tree, ['no surfacing', 'flippers'], [1, 0]))
