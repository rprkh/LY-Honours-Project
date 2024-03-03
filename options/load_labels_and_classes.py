import pickle

def load_labels_and_classes_dict():
    file_path = "assets/labels_and_classes_dict.pickle"

    with open(file_path, "rb") as dict_to_load:
        labels_and_classes_dict = pickle.load(dict_to_load)

    return labels_and_classes_dict