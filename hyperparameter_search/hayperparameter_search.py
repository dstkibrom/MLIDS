import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np
import tensorflow as tf
from utils import dataset_creator, bits_extractor, frame_reader_from_file
import model_creator

all_ids = ['0CF00400', '0CF00300', '18FEF100', '1CFF6F00', '18ECFF00', '18FF8800', '18FF8400',
           '18FEE500', '18F00029', '18FEF200', '18FF7F00', '1CFF7100', '18EBFF00', '18FF8200',
           '18FF8600', '18FEDC00', '1CFF7700', '18FF8900', '18FEDF00', '18FEE900', '18FF8700',
           '18FEE700', '1CFEB300', '18FEC100', '18FEEE00', '18ECFF29', '18EBFF29', '0C000027',
           '0C000F27', '18FEF111', '0CF00203', '0CF00327', '18FF8327', '0C002927', '18FF5027',
           '18F00503', '18FF5127', '18FEED11', '18FEE617', '1CFFAA27', '18EC0027', '18EB0027']

par_bs_size = [64,128,256]
train_min_index=0
train_max_index = 3000000
par_duration = [0.5, 1, 2]
memory = 3000
par_num_layers = [1,2,3]
par_embedding_size = [16,32,64]
par_LSTM_units = [64,128,256]
par_learning_rate=[0.001,0.01,0.1]
opt=tf.keras.optimizers.SGD(learning_rate=par_learning_rate)
no_epochs=100

counter=0
parameters=[]

file = open("minatobus-candump-2019-05-08_030759.log")

_, train_data = frame_reader_from_file.prepare_dataset(file, min_index=train_min_index,
                                                       max_index=train_max_index, arbitration_id=all_ids[0])
val_data = train_data[int(len(train_data) * 0.8):]
train_data = train_data[:int(len(train_data) * 0.8)]

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

for bs_siz in range(len(par_bs_size)):
    for duratio in range(len(par_duration)):
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
                                                        batch_size=par_bs_size[bs_siz], duration=par_duration[duratio], memory_size=memory,
                                                        arbitration_id=all_ids[0])

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
                                                      batch_size=par_bs_size[bs_siz], duration=par_duration[duratio], memory_size=memory,
                                                      arbitration_id=all_ids[0])
        for num_layer in range(len(par_num_layers)):
            for embedding_siz in range(len(par_embedding_size)):
                for LSTM_unit in range(len(par_LSTM_units)):
                    for learning_rat in range(len(par_learning_rate)):
                        counter=counter+1
                        parameters.append([par_bs_size[bs_siz],par_duration[duratio],par_num_layers[num_layer],
                                           par_embedding_size[embedding_siz],par_LSTM_units[LSTM_unit],par_learning_rate[learning_rat]])
                        bs_size=par_bs_size[bs_siz]
                        duration=par_duration[duratio]
                        num_layers=par_num_layers[num_layer]
                        embedding_size=par_embedding_size[embedding_siz]
                        LSTM_units=par_LSTM_units[LSTM_unit]
                        learning_rate=par_learning_rate[learning_rat]

                        model=model_creator.my_model(bs_size,LSTM_units, embedding_size, num_layers)
                        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

                        def loss(labels, logits):
                            return tf.keras.losses.binary_crossentropy(labels, logits, from_logits=True)

                        early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
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

                        history = model.fit(train_data, validation_data=val_data, epochs=no_epochs,verbose=1, callbacks=[early_stopping_cb])
                        file = open("../Arb_id_0/Parameter_settings", "a")
                        file.write(str(parameters)+'---->>>'+ str(history.history['val_loss'])+'\n')
                        file.close()
                        parameters=[]
