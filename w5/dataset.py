from torch.utils.data import Dataset
import torch
import random

class flickrDataset(Dataset):
    def __init__(self, img_features, text_features):
        self.img_data = img_features
        self.text_data = text_features
    
    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, index):
        img, text = self.img_data[index], self.text_data[index]

        return (img,text)
      
# Dataset for triplet loss
class TripletFlickrDatasetImgToTxt(Dataset):
    def __init__(self, flickr_dataset):
        self.dataset = flickr_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get Negative sample
        while True:
            negative_img_id = random.randint(0, len(self.dataset) - 1)
            if negative_img_id != idx:
                break

        while True:
            negative_cap_id = random.randint(0, len(self.dataset) - 1)
            if negative_cap_id != idx:
                break

        image, captions = self.dataset[idx]
        # choose a sample from captions
        caption = captions[random.randint(0, len(captions) - 1)]
        negative_image = self.dataset[negative_img_id][0]
        negative_captions = self.dataset[negative_cap_id][1]
        negative_caption = negative_captions[random.randint(0, len(negative_captions) - 1)]

        # image_triple, caption_triple
        return (image, caption, negative_caption)


class ImageBatch:
    def __init__(self, image):
        assert isinstance(image, torch.FloatTensor), 'Not FloatTensor'
        self.image_batch = image

    def cuda(self):
        self.image_batch = self.image_batch.cuda()
        return self

    def cpu(self):
        self.image_batch = self.image_batch.cpu()
        return self

    def get_batch(self):
        return self.image_batch


class CaptionBatch:
    def __init__(self, caption, caption_length):
        assert isinstance(caption, torch.LongTensor), 'caption Not LongTensor'
        assert isinstance(caption_length, torch.LongTensor), 'caption_length Not LongTensor'
        self.caption_batch = caption
        self.caption_length = caption_length

    def cuda(self):
        self.caption_batch = self.caption_batch.cuda()
        self.caption_length = self.caption_length.cuda()
        return self

    def cpu(self):
        self.caption_batch = self.caption_batch.cpu()
        self.caption_length = self.caption_length.cpu()
        return self

    def get_batch(self):
        return self.caption_batch, self.caption_length


class TripleBatch:
    def __init__(self, anchor, positive, negative):
        self.anchor = anchor
        self.positive = positive
        self.negative = negative

    def cuda(self):
        self.anchor = self.anchor.cuda()
        self.positive = self.positive.cuda()
        self.negative = self.negative.cuda()
        return self

    def cpu(self):
        self.anchor = self.anchor.cpu()
        self.positive = self.positive.cpu()
        self.negative = self.negative.cpu()
        return self

    def get_batch(self):
        return self.anchor.get_batch(), self.positive.get_batch(), self.negative.get_batch()
