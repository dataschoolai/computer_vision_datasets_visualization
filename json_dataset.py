
import fiftyone as fo
#Import libraries for reading and writing files
from pathlib import Path
#Import library for reading JSON files
import json
#Import library to read command line arguments
import argparse


#Define the function create custom dataset format
def get_sample_data(file_path,pred=False):

    data_path =Path(file_path)
    #Create samples list for the dataset
    samples = []
    for path in data_path.glob('labels/*.json'):
        #Read labels from file than assign to annotations
        f = open(path).read()
        annotations = json.loads(f)
        
        # Create sample object
        sample = fo.Sample(path.stem)


        #Create a sample from the file path
        sample = fo.Sample(filepath=data_path/'images'/(path.stem+'.jpg'))
        #Convert detecyions to FiftyOne format
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
    #add samples to dataset
    #Loop through the data path
    for item in data:
        #Check if pred is predictions or ground truth
        
        if item['type']=='predictions':
            dataset.add_samples(get_sample_data(item['path_labels'],True))
        else:
            dataset.add_samples(get_sample_data(item['path_labels'],False))


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
        'type':'predictions'
    },
    {
        'path_labels':'/media/buntuml/DATASET/dataschoolai/yolo_converters/DATASET_JSON',
        'type':'ground_truth'
    }
    ]
    #List of paths to the data
    #Run the app
    app(data)
