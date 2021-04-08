import logging
logging.getLogger('tensorflow').disabled = True
import Version_1.preprocessors.frame_reader_from_file as frame_reader_from_file
import numpy as np
import Version_1.preprocessors.bits_extractor as bits_extractor
import tensorflow as tf
from Version_1 import preprocessors as dataset_creator
import Version_1.preprocessors.model_creator_sigmoid as model_creator_sigmoid
from datetime import datetime
import os

all_ids =['04b1', '00a1', '0430', '02a0', '0130', '0329', '0545', '0370', '05f0', '0316', '0002', '0260', '02b0',
          '05a2', '0440','0140', '0131', '0350', '018f', '0153', '05a0', '0690', '04f0', '043f', '00a0', '02c0', '01f1']



bs_size = 64
duration = 1
memory = 30000
no_epochs=400

conf=[16, 1, 32, 'ITD', 'SGD', 0.1]

embedding_size = conf[0]
num_layers = conf[1]
LSTM_units = conf[2]
lr_schedule_type = conf[3]
optimizer=conf[4]
lr_rate=conf[5]


if lr_schedule_type =='EXP':
    learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_rate,decay_steps=150000,decay_rate=0.96,staircase=True)
else:
    learning_rate= tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=lr_rate, decay_steps=150000, decay_rate=0.96, staircase=True)

if optimizer=='SGD':
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.8, clipvalue=1.0)
else:
    opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=0.8, clipvalue=1.0)

file = open("../datasets/train_data.txt", 'r')

data = frame_reader_from_file.prepare_dataset(file, arbitration_id=all_ids[9])
print(len(data))
train_data = data[:int(len(data) * 0.8)]
val_data = data[int(len(data) * 0.8):]

# bits extraction from packets for training
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(
    np.array(train_data))
# bits extraction from packets for validation
val_bit_0, val_bit_1, val_bit_2, val_bit_3, val_bit_4, val_bit_5, val_bit_6, val_bit_7, val_bit_8, val_bit_9, val_bit_10, \
val_bit_11, val_bit_12, val_bit_13, val_bit_14, val_bit_15, val_bit_16, val_bit_17, val_bit_18, val_bit_19, val_bit_20, \
val_bit_21, val_bit_22, val_bit_23, val_bit_24, val_bit_25, val_bit_26, val_bit_27, val_bit_28, val_bit_29, val_bit_30, \
val_bit_31, val_bit_32, val_bit_33, val_bit_34, val_bit_35, val_bit_36, val_bit_37, val_bit_38, val_bit_39, val_bit_40, \
val_bit_41, val_bit_42, val_bit_43, val_bit_44, val_bit_45, val_bit_46, val_bit_47, val_bit_48, val_bit_49, val_bit_50, \
val_bit_51, val_bit_52, val_bit_53, val_bit_54, val_bit_55, val_bit_56, val_bit_57, val_bit_58, val_bit_59, val_bit_60, \
val_bit_61, val_bit_62, val_bit_63 = bits_extractor.extract_all_bits(np.array(val_data))


train_data = dataset_creator.ready_for_training(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                bit_8, bit_9,
                                                bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16,
                                                bit_17, bit_18,
                                                bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25,
                                                bit_26, bit_27,
                                                bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34,
                                                bit_35, bit_36,
                                                bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43,
                                                bit_44, bit_45,
                                                bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52,
                                                bit_53, bit_54,
                                                bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61,
                                                bit_62, bit_63,
                                                batch_size=bs_size, duration=duration, memory_size=memory,
                                                arbitration_id=all_ids[9])

val_data = dataset_creator.ready_for_training(val_bit_0, val_bit_1, val_bit_2, val_bit_3, val_bit_4,
                                              val_bit_5, val_bit_6, val_bit_7, val_bit_8, val_bit_9,
                                              val_bit_10, val_bit_11, val_bit_12, val_bit_13, val_bit_14,
                                              val_bit_15, val_bit_16, val_bit_17, val_bit_18,
                                              val_bit_19, val_bit_20, val_bit_21, val_bit_22, val_bit_23,
                                              val_bit_24, val_bit_25, val_bit_26, val_bit_27,
                                              val_bit_28, val_bit_29, val_bit_30, val_bit_31, val_bit_32,
                                              val_bit_33, val_bit_34, val_bit_35, val_bit_36,
                                              val_bit_37, val_bit_38, val_bit_39, val_bit_40, val_bit_41,
                                              val_bit_42, val_bit_43, val_bit_44, val_bit_45,
                                              val_bit_46, val_bit_47, val_bit_48, val_bit_49, val_bit_50,
                                              val_bit_51, val_bit_52, val_bit_53, val_bit_54,
                                              val_bit_55, val_bit_56, val_bit_57, val_bit_58, val_bit_59,
                                              val_bit_60, val_bit_61, val_bit_62, val_bit_63,
                                              batch_size=bs_size, duration=duration, memory_size=memory,
                                              arbitration_id=all_ids[9])

model= model_creator_sigmoid.my_model(bs_size, LSTM_units, embedding_size, num_layers)

def loss(labels, logits):
    return tf.keras.losses.binary_crossentropy(labels, logits)


early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb= tf.keras.callbacks.TensorBoard(log_dir=logdir)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
model.compile(optimizer=opt,
                   loss=[loss, loss, loss, loss, loss, loss, loss, loss, loss, loss, loss, loss,
                         loss, loss, loss,
                         loss, loss, loss, loss, loss, loss, loss, loss, loss, loss, loss, loss,
                         loss, loss, loss,
                         loss, loss, loss, loss, loss, loss, loss, loss, loss, loss, loss, loss,
                         loss,
                         loss, loss, loss, loss, loss, loss, loss, loss, loss, loss, loss, loss,
                         loss,
                         loss, loss, loss, loss, loss, loss, loss, loss])

model.fit(train_data, validation_data=val_data, epochs=no_epochs,verbose=1, callbacks=[early_stopping_cb,tensorboard_cb, checkpoint_callback])
