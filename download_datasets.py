import tensorflow_datasets as tfds
import resource
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",    type=str,   help="Set the dataset download directory",
                    default='./datasets/')
parser.add_argument("--train_dataset",  type=str,   help="Set the dataset to be used for training (voc | coco)",
                    default='voc')

args = parser.parse_args()

if __name__ == '__main__':
    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))

    os.makedirs(args.dataset_dir, exist_ok=True)

    if args.train_dataset == 'voc':
        train_pascal_12 = tfds.load('voc/2012', data_dir=args.dataset_dir, split='train')
        valid_train_12 = tfds.load('voc/2012', data_dir=args.dataset_dir, split='validation')

        train_pascal_07 = tfds.load("voc", data_dir=args.dataset_dir, split='train')
        valid_train_07 = tfds.load("voc", data_dir=args.dataset_dir, split='validation')

        test_data = tfds.load('voc', data_dir=args.dataset_dir, split='test')

        train_data = train_pascal_07.concatenate(train_pascal_12)
        valid_data = valid_train_07.concatenate(valid_train_12)

        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
        number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
        
        print("Nuber of train dataset = {0}".format(number_train))
        print("Nuber of validation dataset = {0}".format(number_valid))
        print("Nuber of test dataset = {0}".format(number_test))


    elif args.train_dataset == 'coco' :
        train_data = tfds.load('coco/2017', data_dir=args.dataset_dir, split='train')
        valid_data = tfds.load('coco/2017', data_dir=args.dataset_dir, split='validation')
        test_data = tfds.load('coco/2017', data_dir=args.dataset_dir, split='test')

        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()      
        number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
        
        print("Nuber of train dataset = {0}".format(number_train))
        print("Nuber of validation dataset = {0}".format(number_valid))
        print("Nuber of test dataset = {0}".format(number_test))
    
    else:
        raise print("check train_dataset name. dataset_download.py can only download voc or coco datasets")
