from os import listdir
from os.path import isfile, join

DATASET_DIR = "datasets/bonn/"
onlyfiles = [f for f in listdir(DATASET_DIR) if isfile(join(DATASET_DIR, f))]

print("Found " + str(len(onlyfiles)) + " files in directory: " + DATASET_DIR)

content = []
for i in range(0, len(onlyfiles)):
    with open(DATASET_DIR + onlyfiles[i], 'r') as content_file:
        list.append(content, content_file.read())

print(len(content))
