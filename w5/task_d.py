import torch
import os
import json
import numpy as np
from transformers import BertTokenizer, BertModel

data_path = '/home/group01/mcv/datasets/Flickr30k'
img_features_file = '{}/vgg_feats.mat'.format(data_path)
text_features_file = '{}/fasttext_feats.npy'.format(data_path)
output_path = "./results/task_d"


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

    #bert_model = bert_model.to(device)

    #TRAIN DATA
    train_TextFeatures = []
    with torch.no_grad():
        bert_model.eval()

        for i, key in enumerate(train_data):
            sentences = []
            for j, sentence in enumerate(key['sentences']):
                #print("{}, {}".format(i,sentence['raw']))
                x = sentence['raw']
                x = bert_tokenizer(x, return_tensors="pt")
                x = bert_model(**x)['last_hidden_state']
                #print(x)
                sentences.append(np.array(x.squeeze().numpy()))
            train_TextFeatures.append(np.array(sentences))
            '''if i == 10:
                break'''

    train_TextFeatures = np.array(train_TextFeatures)

    #Save data
    os.makedirs(output_path, exist_ok=True)
    with open('{}/train_Textfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, train_TextFeatures)       
    #print(train_TextFeatures)
    #print('shape: {}'.format(train_TextFeatures.shape))
    print('Text features from Train dataset saved.')

    # VALIDATION DATA
    val_TextFeatures = []
    with torch.no_grad():
        bert_model.eval()

        for i, key in enumerate(val_data):
            sentences = []
            for j, sentence in enumerate(key['sentences']):
                #print("{}, {}".format(i,sentence['raw']))
                x = sentence['raw']
                x = bert_tokenizer(x, return_tensors="pt")
                x = bert_model(**x)['last_hidden_state']
                #print(x)
                sentences.append(np.array(x.squeeze().numpy()))
            val_TextFeatures.append(np.array(sentences))
            '''if i == 10:
                break'''

    val_TextFeatures = np.array(val_TextFeatures)

    #Save data
    os.makedirs(output_path, exist_ok=True)
    with open('{}/val_Textfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, val_TextFeatures)       
    #print(train_TextFeatures)
    #print('shape: {}'.format(train_TextFeatures.shape))
    print('Text features from Validation dataset saved.')

    # TEST DATA
    test_TextFeatures = []
    with torch.no_grad():
        bert_model.eval()

        for i, key in enumerate(test_data):
            sentences = []
            for j, sentence in enumerate(key['sentences']):
                #print("{}, {}".format(i,sentence['raw']))
                x = sentence['raw']
                x = bert_tokenizer(x, return_tensors="pt")
                x = bert_model(**x)['last_hidden_state']
                #print(x)
                sentences.append(np.array(x.squeeze().numpy()))
            test_TextFeatures.append(np.array(sentences))
            '''if i == 10:
                break'''

    test_TextFeatures = np.array(test_TextFeatures)

    #Save data
    os.makedirs(output_path, exist_ok=True)
    with open('{}/test_Textfeatures.npy'.format(output_path), "wb") as f:
        np.save(f, test_TextFeatures)       
    #print(train_TextFeatures)
    #print('shape: {}'.format(train_TextFeatures.shape))
    print('Text features from Test dataset saved.')