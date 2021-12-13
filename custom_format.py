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
            confidence = ann['score']
            #Normalize bounding box
            x1 = bbox[0]/width
            y1 = bbox[1]/height
            w = bbox[2]/width
            h = bbox[3]/height
        
            #Create detection object
            detection = fo.Detection(label=label,bounding_box=[x1,y1,w,h],confidence=confidence)
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

#Define the function to parse the command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Create custom dataset format')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to the dataset')
    #Get predictions flag default value is False
    parser.add_argument('--pred', type=bool, default=False,
                        help='Flag to get predictions')
    return parser.parse_args()


#Define the function luanch the app

def app(samples):
    # path='DATA/CONCRETE_DATASET_JSON'
    #Create dataset object
    dataset = fo.Dataset('Custom-data')
    #add samples to dataset
    dataset.add_samples(samples)


    #Launch the session 
    session = fo.launch_app()
    #Add dataset to session
    session.dataset=dataset
    #Wait session to finish
    session.wait()


#Run the main function
if __name__ == '__main__':
    args = parse_args()
    samples = get_sample_data(args.path,pred=args.pred)
    #Create custom dataset format
    # fo.Dataset(samples=samples).save('custom_format.json')
    app(samples)
