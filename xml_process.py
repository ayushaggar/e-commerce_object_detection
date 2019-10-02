import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
np.random.seed(1)

def main():
    # list file in folder
    files = os.listdir(os.getcwd() + '/annotations')
    xml_files = [f for f in files if f[-3:] == 'xml']
    for idx, val in enumerate(xml_files):
        file_name = os.getcwd() + '/annotations/' + val
        tree = ET.parse(file_name)
        root = tree.getroot()
        elements = []
        values = []
        [elements.append(elem.tag) for elem in root.iter()]
        [values.append(elem.text) for elem in root.iter()]

        if 'name' in elements:
            if idx == 0:
                input_df  = pd.DataFrame(columns = elements)
            new_entry = dict(zip(elements, values))
            input_df.loc[len(input_df)] = new_entry
    
    # drop unused columns
    input_df = input_df.drop(['depth','annotation', 'source', 'size', 'object', 'bndbox', 'difficult', 'truncated', 'pose', 'segmented','database','path','folder'], 1)
    input_df.rename(
        columns={
            'name': 'class'},
        inplace=True)
    input_df.to_csv('data/images_labels.csv', index=False)
    msk = np.random.rand(len(input_df)) < 0.8
    train_df = input_df[msk]
    test_df = input_df[~msk]
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
main()