
from os import pread
import fiftyone as fo
#Import libraries for reading and writing files
from pathlib import Path
#Import library for reading JSON files
import json
#Import library to read command line arguments
import argparse

#Define the function create fiftyone detection format
def get_detection_data(annotations:dict,pred=False):
    #Create detections list for the dataset
    detections = []
    #read from annotations image width and height
    width = annotations['images'][0]['width']
    height = annotations['images'][0]['height']
    
    for ann in annotations['annotations']:
        label = '103'
        # Bounding box coordinates should be relative values
        # in [0, 1] in the following format:
        # [top-left-x, top-left-y, width, height]
        #Read bounding box from annotation
        bbox = ann['bbox']
        #Get confidence from annotation
        confidence = ann.get('score',False)
        #Normalize bounding box
        x1 = bbox[0]/width
        y1 = bbox[1]/height
        w = bbox[2]/width
        h = bbox[3]/height
    
        #Create detection object
        if pred:
            detection = fo.Detection(label=label,bounding_box=[x1,y1,w,h],confidence=confidence)
        else:
            detection = fo.Detection(label=label,bounding_box=[x1,y1,w,h])
        #Append detection to detections list
        detections.append(detection)

    return detections

#Define the function create custom dataset format
def get_sample_data(images_path:Path,labels_path:list):
    # label_path = Path(labels_path[0]['path_labels'])/'labels'
    # pred = labels_path[0]['pred']
    #Create samples list for the dataset
    samples = []
    for path in images_path.glob('*.jpg'):
        #Read labels from file than assign to annotations
        # sample = fo.Sample(path.stem)
        sample = fo.Sample(filepath=path)
        for label_path in labels_path:
            label =Path(label_path['path_labels'])/'labels' / path.stem 

            pred = label_path['pred']
            #Add extension to label
            label = label.with_suffix('.json')
            #Check if file is not exists
            

            if not label.exists():
                continue
            
            f = open(label).read()
            annotations = json.loads(f)
            
            # Create sample object


            #Create a sample from the file path
            # sample = fo.Sample(filepath=path)
            #Loop through labels

            #Convert detecyions to FiftyOne format
            detections = get_detection_data(annotations,pred)
            #Assign detections to sample
            if pred:
                sample['predictions'] = fo.Detections(detections=detections)
            else:
                sample['ground_truth'] = fo.Detections(detections=detections)

            #append sample to samples list
        samples.append(sample)
    return samples

#Define the function luanch the app

def app(data_path:dict):
    # path='DATA/CONCRETE_DATASET_JSON'
    #Create dataset object
    dataset = fo.Dataset('Custom-data')
    #Define the images path
    images_path = Path(data_path[1]['path_labels'])/'images'
    # labels_path = Path(data_path[1]['path_labels'])/'labels'

    dataset.add_samples(get_sample_data(images_path=images_path,labels_path=data_path))


    #Launch the session 
    session = fo.launch_app()
    #Add dataset to session
    session.dataset=dataset
    #Wait session to finish
    session.wait()


#Run the main function 
if __name__ == '__main__':
    #Define the directory path to the dataset

    data =[{
        'path_labels':'/media/buntuml/DATASET/dataschoolai/yolov5_inference/dataset_json',
        'pred': True
    },
    {
        'path_labels':'/media/buntuml/DATASET/dataschoolai/yolo_converters/DATASET_JSON',
        'pred':False
    }
    ]
    #List of paths to the data
    #Run the app
    app(data)
