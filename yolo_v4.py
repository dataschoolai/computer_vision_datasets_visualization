import fiftyone as fo

name = "my-dataset"
data_path = "/media/buntuml/DATASET/DAMAGEAI/REPORT/deepeye_mutli_damage/damage_multiclass"
labels_path = "/media/buntuml/DATASET/DAMAGEAI/REPORT/deepeye_mutli_damage/damage_multiclass"
classes = [
    "spallcross",
"pothole",
"spalling",
"patchdamage"
]

# Create the dataset
# Import dataset by explicitly providing paths to the source media and labels
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.YOLOv4Dataset,
    data_path=data_path,
    labels_path=labels_path,
    classes=classes,
    name=name,
)
# View summary info about the dataset
print(dataset)

# Print the first few samples in the dataset
print(dataset.head())

#Launch the session 
session = fo.launch_app()
#Add dataset to session
session.dataset=dataset
#Wait session to finish
session.wait()

