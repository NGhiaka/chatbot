import os
import tensorflow as tf
from models.transformer import transformer
from core.dataset import Dataset
from utils.preprocessing import preprocess_senctence
from core.config import cfg

tf.keras.backend.clear_session()

dataset = Dataset(cfg.TRAIN.MAX_LENGTH, cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.BUFFER_SIZE, cfg.TRAIN.DATA_PATH)
model = transformer(
    vocab_size=dataset.vocab_size,
    num_layers=cfg.TRAIN.NUM_LAYERS,
    units=cfg.TRAIN.UNITS,
    d_model=cfg.TRAIN.D_MODEL,
    num_heads=cfg.TRAIN.NUM_HEADS,
    dropout=cfg.TRAIN.DROPOUT)

model.load_weights(cfg.TRAIN.CHECKPOINT_PATH)

def evaluate(sentence):
	sentence = preprocess_senctence(sentence)

	sentence = tf.expand_dims(
		dataset.start_token + dataset.tokenizer.encode(sentence) + dataset.start_token, axis=0)

	output = tf.expand_dims(dataset.start_token, 0)

	for i in range(cfg.TRAIN.MAX_LENGTH):
		predictions = model(inputs=[sentence, output], training=False)

		# select the last word from the seq_len dimension
		predictions = predictions[:, -1:, :]
		predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

		#return the result if predicted_id is equal to the end token
		if tf.equal(predicted_id, dataset.start_token[0]):
			break

		output = tf.concat([output, predicted_id], axis=-1)

	return tf.squeeze(output, axis=0)

def predict(sentence):
	prediction = evaluate(sentence)

	predicted_sentence = dataset.tokenizer.decode(
		[i for i in prediction if i < dataset.tokenizer.vocab_size])

	# print('Input: {}'.format(sentence))
	# print('Output: {}'.format(predicted_sentence))

	return predicted_sentence

# response = predict('whats the balance on my account')