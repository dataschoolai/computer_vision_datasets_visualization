import fiftyone as fo

name = "my-dataset"
dataset_dir = "/media/buntuml/DATASET/DAMAGEAI/REPORT/efflorescence_1/CONCRETE_DATASET_YOLOv5"
# dataset_dir = '/home/buntuml/Documents/dataset/North American Mushrooms.v1-416x416 (1).yolov5pytorch'
# label = '/home/buntuml/github/damage.ai/CrackDetectionDataset/CONCRETE_DATASET_YOLOv5/dataset.yml'
session = fo.launch_app()
# Create the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    # labels_path=label,
    name=name,
)
print(type(dataset))
# View summary info about the dataset
session.dataset = dataset
print(dataset)
session.wait()

# Print the first few samples in the dataset
# print(dataset.head())
