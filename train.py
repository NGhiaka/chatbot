import os
import tensorflow as tf
from models.transformer import transformer
from core.dataset import Dataset
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

def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, cfg.TRAIN.MAX_LENGTH - 1))
  
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(cfg.TRAIN.D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
  # ensure labels have shape (batch_size, MAX_LENGTH - 1)
  y_true = tf.reshape(y_true, shape=(-1, cfg.TRAIN.MAX_LENGTH - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

##Save model
# checkpoint_dir = os.path.dirname(cfg.TRAIN.CHECKPOINT_PATH)
# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cfg.TRAIN.CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)
stop_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, mode='auto')
model.fit(dataset(), epochs=cfg.TRAIN.EPOCHS, callbacks=[cp_callback, stop_callback])