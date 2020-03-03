from datetime import datetime
import json
import os
import argparse

import pandas as pd
import boto3
from PIL import Image



def getIndex(df, index, value):
    return df.loc[df[index] == value].index.item()
    
def getRow(df, column, value):
    return df.loc[df[column] == value]

if __name__ == '__main__':
    # Parser for command line arguments
    parser = argparse.ArgumentParser(description='Construct a manifest file to retrain AWS Rekognition on the OpenImages dataset.')
    parser.add_argument('classes', type=str, help='File containing class descriptions of the boxes')
    parser.add_argument('bboxes', type=str, help='File containing all the bounding boxes')
    parser.add_argument('s3_bucket', type=str, help='Name of the S3 bucket')
    parser.add_argument('s3_bucket_folder', type=str, help='Name of the folder (prefix) on the S3 bucket')
    parser.add_argument('output_path', type=str, help='Path for the output manifest file.')
    args = parser.parse_args()

    # Opens the output file
    manifest_file = open(args.output_path, 'w+')

    # Those are utility mapping for converting one format to the other
    class_mapping_df = pd.read_csv(args.classes, header=None)
    class_map = {index:row[1] for index, row in class_mapping_df.iterrows()}
    inverted_class_map = {v: k for k, v in class_map.items()}
    rawlabel2id = {row[0]:index for index, row in class_mapping_df.iterrows()}
    label2id = {row[0]:index for index, row in class_mapping_df.iterrows()}
    
    annotations = pd.read_csv(args.bbox)

    s3 = boto3.client('s3')

    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=args.s3_bucket, Prefix=args.s3_bucket_folder)

    file = {}
    for file_id in os.listdir('train'):
        file['Key'] = f'train/{file_id}'
        if file['Key'].endswith('.jpg'):
            with Image.open(file['Key']) as im:
                width_image, height_image = im.size
            
            # Construct the manifest entry for this picture
            manifest_entry = {}
            manifest_entry['source-ref'] = f"s3://{args.s3_bucket_folder}/{file['Key']}"
            manifest_entry['bounding-box'] = {}
            manifest_entry['bounding-box']['image_size'] = [
                {
                    'width': width_image,
                    'height': height_image,
                    'depth': 3
                }
            ]
            manifest_entry['bounding-box']['annotations'] = []
            manifest_entry['bounding-box-metadata'] = {
                'objects': [],
                'class-map': {},
                'type': 'groundtruth/object-detection',
                'human-annotated': 'yes',
                "creation-date": datetime.now().isoformat()
            }

            # Get the annotations corresponding to that image
            image_id = file['Key'].split('/')[1].split('.')[0]
            image_annotations = getRow(annotations, 'ImageID', image_id)

            for index, row in image_annotations.iterrows():
                # We have to convert the annotations
                xmin, xmax, ymin, ymax = row['XMin'], row['XMax'], row['YMin'], row['YMax']
                coded_label = row['LabelName']

                class_id = rawlabel2id[coded_label]
                left = int(xmin * width_image)
                top = int(ymin * height_image)
                width = int((xmax - xmin) * width_image)
                height = int((ymax - ymin) * height_image)

                manifest_entry['bounding-box']['annotations'].append(
                    {
                        'class_id': class_id,
                        'left': left,
                        'top': top,
                        'width': width,
                        'height': height
                    }
                )
                manifest_entry['bounding-box-metadata']['objects'].append({'confidence':1})
                manifest_entry['bounding-box-metadata']['class-map'][class_id] = class_map[class_id]
            manifest_file.write(json.dumps(manifest_entry) + '\n')
    manifest_file.close()