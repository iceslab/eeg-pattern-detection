# Imports
from os import listdir
from os.path import isfile, join

# Default dataset directory
DATASET_DIR = "datasets/bonn/"
SUBSAMPLE_DIR = "datasets/bonn/subsample"


# Loads files from given dataset directory
def load_files(dataset_directory):
    print("Reading files...")
    onlyfiles = [f for f in listdir(dataset_directory) if isfile(join(dataset_directory, f))]

    print("Found " + str(len(onlyfiles)) + " files in directory: " + dataset_directory)

    content = []
    labels = []
    for i in range(0, len(onlyfiles)):
        with open(dataset_directory + onlyfiles[i], 'r') as content_file:
            read_content = content_file.read().splitlines()
            list.append(content, read_content)
            list.append(labels, onlyfiles[i][:1])

    # Changing string values to numbers
    labels_set = set(labels)
    labels_dictionary = {}
    index = 0
    for set_element in labels_set:
        labels_dictionary[set_element] = index
        index += 1

    for i in range(0, len(labels)):
        labels[i] = int(labels_dictionary.get(labels[i]))
    # print(len(content))
    # plt.plot(content[0])
    # plt.show()
    print("Read files")
    return content, labels


def subsample_data(content, labels):
    print("Subsampling data...")
    subsample_len = 178

    for idx in range(0, len(content)):
        print("File " + str(idx + 1) + "/" + str(len(content)))
        for i in range(0, len(content[idx]) - subsample_len):
            file = open(SUBSAMPLE_DIR + "/" + str(labels[idx]) + "_" + str(i) + ".txt", "w")
            for el in content[idx][i:i+subsample_len]:
                file.write(el + "\n")
            file.close()

    print("Subsampled data")


if __name__ == "__main__":
    # Load training and eval data
    cont, lbl = load_files(DATASET_DIR)
    subsample_data(cont, lbl)
