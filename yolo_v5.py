import fiftyone as fo
import argparse
# Define the function create dataset
def get_dataset(path):
 
    # Dataset name
    name = "yolo_v5_dataset"
    # The splits to load
    splits = ["train", "val"]
    dataset_dir = path

    # Load the dataset, using tags to mark the samples in each split
    dataset = fo.Dataset(name)
    for split in splits:
        dataset.add_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.YOLOv5Dataset,
            split=split,
            tags=split,
    )
    return dataset

#Define the function to parse the command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Create a dataset")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the directory with images",
    )
    return parser.parse_args()

#Define the function to luanch the app
def app(dataset):
    #Get the dataset
    dataset = get_dataset(args.path)
    #lunch the session
    session = fo.launch_app()
    # View summary info about the dataset

    session.dataset = dataset
    print(dataset)
    session.wait()
#Run the main function
if __name__ == "__main__":
    args = parse_args()
    app(args.path)


