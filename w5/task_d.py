import torch
import os
import json
import numpy as np
from transformers import BertTokenizer, BertModel
from time import time

data_path = '/home/group01/mcv/datasets/Flickr30k'
output_path = "./results/task_d"
os.makedirs(output_path, exist_ok=True)

def extract_features(data):
    TextFeatures = []
    with torch.no_grad():
        bert_model.eval()
        start_global_time = time()
        for i, key in enumerate(data):
            sentences = []
            start = time()
            for j, sentence in enumerate(key['sentences']):
                #print("{}, {}".format(i,sentence['raw']))
                x = sentence['raw']
                x = bert_tokenizer(x, return_tensors="pt")
                x = x.to(device)
                x = bert_model(**x)['last_hidden_state']
                #print(x)
                x = x.to("cpu")
                sentences.append(np.array(x.squeeze().numpy()))
            TextFeatures.append(sentences)
            end = time() - start
            print('Time spent in image id {}: {:.3f} seconds'.format(i,end))
            '''if i == 100:
                break'''
        end_global_time = time() - start_global_time
        print('The evaluation has taken {:.3f} seconds.'.format(end_global_time))

    #train_TextFeatures = np.array(train_TextFeatures)
    return TextFeatures

if __name__ == '__main__':

    with open(f'{data_path}/train.json') as f:
        train_data = json.load(f)

    with open(f'{data_path}/val.json') as f:
        val_data = json.load(f)

    with open(f'{data_path}/test.json') as f:
        test_data = json.load(f)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Initializing a model from the bert-base-uncased style configuration
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_model = bert_model.to(device)

    #TRAIN DATA
    print('Start extracting Text features from Train dataset...')
    train_TextFeatures = extract_features(train_data)

    #Save data
    with open('{}/train_Textfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, train_TextFeatures)       
    print('Text features from Train dataset saved.')


    # VALIDATION DATA  
    print('Start extracting Text features from Validation dataset...')
    val_TextFeatures = extract_features(val_data)

    #Save data
    with open('{}/val_Textfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, val_TextFeatures)       
    print('Text features from Validation dataset saved.')


    print('Start extracting Text features from Test dataset...')
    # TEST DATA
    test_TextFeatures = extract_features(test_data)

    #Save data
    with open('{}/test_Textfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, test_TextFeatures)       
    print('Text features from Test dataset saved.')
    