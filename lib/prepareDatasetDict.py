'''
it will take a manifest file as input. and produce datasetDict as output.
'''
import json
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm
from datasets import Dataset, DatasetDict


COLUMNS = ['file', 'audio', 'text']


def createDatasetDict(train_manifest_path, val_manifest_path, test_manifest_path, data_location = './all_audio_in_wav'):
    trainDF = pd.DataFrame(columns=COLUMNS)
    print(f'READING TRAIN MANIFEST: {train_manifest_path}')
    with open (train_manifest_path, 'r') as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines)):
            try:
                jsonObj = json.loads(line)
                path = jsonObj["audio_filepath"]
                path = f"{data_location}/{path.split(sep = '/')[-1]}"
                text = jsonObj["text"]
                samplerate, audio = wavfile.read(path)
                trainDF.loc[i] = [path, audio, text]
            except:
                print('error occurred. but you can ignore it.')
    









    valDF = pd.DataFrame(columns=COLUMNS)
    print(f'READING VAL MANIFEST: {val_manifest_path}')
    with open (val_manifest_path, 'r') as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines)):
            try:
                jsonObj = json.loads(line)
                path = jsonObj["audio_filepath"]
                path = f"{data_location}/{path.split(sep = '/')[-1]}"
                text = jsonObj["text"]
                samplerate, audio = wavfile.read(path)
                valDF.loc[i] = [path, audio, text]
            except:
                print('error occurred. but you can ignore it.')

    



    testDF = pd.DataFrame(columns=COLUMNS)
    print(f'READING TEST MANIFEST: {test_manifest_path}')
    with open (test_manifest_path, 'r') as f:
        lines = f.readlines()
        for i, line in tqdm(enumerate(lines)):
            try:
                jsonObj = json.loads(line)
                path = jsonObj["audio_filepath"]
                path = f"{data_location}/{path.split(sep = '/')[-1]}"
                text = jsonObj["text"]
                samplerate, audio = wavfile.read(path)
                testDF.loc[i] = [path, audio, text]
            except:
                print('error occurred. but you can ignore it.')



    train_dataset = Dataset.from_dict(trainDF)
    val_dataset = Dataset.from_dict(valDF)
    test_dataset = Dataset.from_dict(testDF)
    return DatasetDict({"train":train_dataset,"val":val_dataset,"test":test_dataset})