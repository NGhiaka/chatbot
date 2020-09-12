import os
import re
import tensorflow as tf
import tensorflow_datasets as tfds
from core.config import cfg
from utils.io import load_conversation


class Dataset(object):
    def __init__(self, max_length, batch_size, buffer_size, file_path):
        self.max_length = max_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.questions, self.answers = load_conversation(file_path)

        self.tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            self.questions + self.answers, target_vocab_size=2**13)
        
        self.start_token, self.end_token = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]

        self.vocab_size = self.tokenizer.vocab_size + 2

    def tokenize_and_filter(self):
        tokenized_inputs, tokenized_outputs = [], []

        for (sentence1, sentence2) in zip(self.questions, self.answers):
            # tokenize sentente
            sentence1 = self.start_token + self.tokenizer.encode(sentence1) + self.end_token
            sentence2 = self.start_token + self.tokenizer.encode(sentence2) + self.end_token

            if len(sentence1) <= self.max_length and len(sentence2) < self.max_length:
                tokenized_inputs.append(sentence1)
                tokenized_outputs.append(sentence2)
        #pad tokenized sentences
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=self.max_length, padding="post")
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=self.max_length, padding="post")

        return tokenized_inputs, tokenized_outputs
    def __call__(self):
        questions, answers = self.tokenize_and_filter()
        dataset  = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': questions,
                'dec_inputs': answers[:, :-1]
            },
            {
                'outputs': answers[:, 1:]
            }
        ))
        dataset = dataset.cache()
        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

# file = "data/communication.txt"
# dataset = Dataset(40, 2, 20000, file)
# d = dataset()
# for (batch, (question, answer)) in enumerate(d):
#     print(question)
#     print(answer)
# # questions, answers = dataset.tokenize_and_filter()
# # print(d)
# print(next(iter(d)))

