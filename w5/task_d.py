import torch
import os
import json
import numpy as np
from transformers import BertTokenizer, BertModel

data_path = '/home/group01/mcv/datasets/Flickr30k'
img_features_file = '{}/vgg_feats.mat'.format(data_path)
text_features_file = '{}/fasttext_feats.npy'.format(data_path)
output_path = "./results/task_d/"


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

    train_TextFeatures = np.empty((len(train_data), 5, 300))
    with torch.no_grad():
        bert_model.eval()
        
        for i, key in enumerate(train_data):
            for j, sentence in key['sentences']:
                #print("{}, {}".format(i,sentence['raw']))
                x = sentence['raw']
                x = bert_tokenizer(x, return_tensors="pt")
                x = bert_model(**x)['last_hidden_state']
                print(x)
                train_TextFeatures[i,j, :] = x.squeeze().numpy()
            if i == 10:
                break
        
    print(train_TextFeatures)
    
    '''state_dict = [image_model.state_dict(), text_model.state_dict()]
    model_folder = str(output_path + "/models")
    os.makedirs(model_folder, exist_ok=True)
    torch.save(state_dict, '{0}/Image2Text_weights.pth'.format(model_folder))'''
