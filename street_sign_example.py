from my_utils import split_data, order_test_set
from deeplearning_models import streetsigns_model
           
if __name__ == "__main__":
    if False:
        path_to_data = "D:\\Dom\\Documents\\Project\\FreeCodeCamp\\Tensorflow\\data\\Train"
        path_to_save_train = "D:\\Dom\\Documents\\Project\\FreeCodeCamp\\Tensorflow\\data\\training_data\\train"
        path_to_save_val = "D:\\Dom\\Documents\\Project\\FreeCodeCamp\\Tensorflow\\data\\training_data\\val"
        split_data(path_to_data, path_to_save_train, path_to_save_val)
    
    path_to_images = "D:\Dom\Documents\Project\FreeCodeCamp\Tensorflow\data\Test"
    path_to_csv = "D:\Dom\Documents\Project\FreeCodeCamp\Tensorflow\data\Test.csv"
    order_test_set(path_to_images, path_to_csv)