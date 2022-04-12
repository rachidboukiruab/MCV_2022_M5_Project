import logging
import math
from datetime import datetime

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers import models
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch import nn
from torch.utils.data import DataLoader

model_name = 'bert-base-uncased'
train_batch_size = 16
num_epochs = 1
model_save_path = './output/training_stsbenchmark_continue_training-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

word_embedding_model = models.Transformer(model_name, max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=512,
                           activation_function=nn.Tanh())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
# TODO train $ val readers with InputExample
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.ContrastiveLoss(model=model)

# Development set: Measure correlation between cosine score and gold labels
logging.info("Read evaluation dev dataset")
evaluator = BinaryClassificationEvaluator.from_input_examples(val_samples, name='contrast-dev')

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)
