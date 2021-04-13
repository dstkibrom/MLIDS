import tensorflow as tf
from model import model_creator_sigmoid
import logging
import os

logging.getLogger('tensorflow').disabled = True

all_ids = ['0CF00400', '0CF00300', '18FEF100', '1CFF6F00', '18ECFF00', '18FF8800', '18FF8400',
           '18FEE500', '18F00029', '18FEF200', '18FF7F00', '1CFF7100', '18EBFF00', '18FF8200',
           '18FF8600', '18FEDC00', '1CFF7700', '18FF8900', '18FEDF00', '18FEE900', '18FF8700',
           '18FEE700', '1CFEB300', '18FEC100', '18FEEE00', '18ECFF29', '18EBFF29', '0C000027',
           '0C000F27', '18FEF111', '0CF00203', '0CF00327', '18FF8327', '0C002927', '18FF5027',
           '18F00503', '18FF5127', '18FEED11', '18FEE617', '1CFFAA27', '18EC0027', '18EB0027']


def ready_for_training(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12,
                       bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25,
                       bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, bit_37, bit_38,
                       bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52,
                       bit_53, bit_54, bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,arb_id):

    batch_size = 1
    LSTM_units = 32
    embedding_size = 16
    num_layers = 1
    checkpoint_dir = '../trained_models/'+str(all_ids.index(arb_id))+'/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
    model = model_creator_sigmoid.my_model(batch_size, LSTM_units, embedding_size, num_layers)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    whole_sequence_loss = []
    for i in range(bit_0.shape[0]):
        bit_0_input, bit_0_output = bit_0[i, :-1].numpy().tolist(), bit_0[i, 1:].numpy().tolist()
        bit_1_input, bit_1_output = bit_1[i, :-1].numpy().tolist(), bit_1[i, 1:].numpy().tolist()
        bit_2_input, bit_2_output = bit_2[i, :-1].numpy().tolist(), bit_2[i, 1:].numpy().tolist()
        bit_3_input, bit_3_output = bit_3[i, :-1].numpy().tolist(), bit_3[i, 1:].numpy().tolist()
        bit_4_input, bit_4_output = bit_4[i, :-1].numpy().tolist(), bit_4[i, 1:].numpy().tolist()
        bit_5_input, bit_5_output = bit_5[i, :-1].numpy().tolist(), bit_5[i, 1:].numpy().tolist()
        bit_6_input, bit_6_output = bit_6[i, :-1].numpy().tolist(), bit_6[i, 1:].numpy().tolist()
        bit_7_input, bit_7_output = bit_7[i, :-1].numpy().tolist(), bit_7[i, 1:].numpy().tolist()
        bit_8_input, bit_8_output = bit_8[i, :-1].numpy().tolist(), bit_8[i, 1:].numpy().tolist()
        bit_9_input, bit_9_output = bit_9[i, :-1].numpy().tolist(), bit_9[i, 1:].numpy().tolist()
        bit_10_input, bit_10_output = bit_10[i, :-1].numpy().tolist(), bit_10[i, 1:].numpy().tolist()
        bit_11_input, bit_11_output = bit_11[i, :-1].numpy().tolist(), bit_11[i, 1:].numpy().tolist()
        bit_12_input, bit_12_output = bit_12[i, :-1].numpy().tolist(), bit_12[i, 1:].numpy().tolist()
        bit_13_input, bit_13_output = bit_13[i, :-1].numpy().tolist(), bit_13[i, 1:].numpy().tolist()
        bit_14_input, bit_14_output = bit_14[i, :-1].numpy().tolist(), bit_14[i, 1:].numpy().tolist()
        bit_15_input, bit_15_output = bit_15[i, :-1].numpy().tolist(), bit_15[i, 1:].numpy().tolist()
        bit_16_input, bit_16_output = bit_16[i, :-1].numpy().tolist(), bit_16[i, 1:].numpy().tolist()
        bit_17_input, bit_17_output = bit_17[i, :-1].numpy().tolist(), bit_17[i, 1:].numpy().tolist()
        bit_18_input, bit_18_output = bit_18[i, :-1].numpy().tolist(), bit_18[i, 1:].numpy().tolist()
        bit_19_input, bit_19_output = bit_19[i, :-1].numpy().tolist(), bit_19[i, 1:].numpy().tolist()
        bit_20_input, bit_20_output = bit_20[i, :-1].numpy().tolist(), bit_20[i, 1:].numpy().tolist()
        bit_21_input, bit_21_output = bit_21[i, :-1].numpy().tolist(), bit_21[i, 1:].numpy().tolist()
        bit_22_input, bit_22_output = bit_22[i, :-1].numpy().tolist(), bit_22[i, 1:].numpy().tolist()
        bit_23_input, bit_23_output = bit_23[i, :-1].numpy().tolist(), bit_23[i, 1:].numpy().tolist()
        bit_24_input, bit_24_output = bit_24[i, :-1].numpy().tolist(), bit_24[i, 1:].numpy().tolist()
        bit_25_input, bit_25_output = bit_25[i, :-1].numpy().tolist(), bit_25[i, 1:].numpy().tolist()
        bit_26_input, bit_26_output = bit_26[i, :-1].numpy().tolist(), bit_26[i, 1:].numpy().tolist()
        bit_27_input, bit_27_output = bit_27[i, :-1].numpy().tolist(), bit_27[i, 1:].numpy().tolist()
        bit_28_input, bit_28_output = bit_28[i, :-1].numpy().tolist(), bit_28[i, 1:].numpy().tolist()
        bit_29_input, bit_29_output = bit_29[i, :-1].numpy().tolist(), bit_29[i, 1:].numpy().tolist()
        bit_30_input, bit_30_output = bit_30[i, :-1].numpy().tolist(), bit_30[i, 1:].numpy().tolist()
        bit_31_input, bit_31_output = bit_31[i, :-1].numpy().tolist(), bit_31[i, 1:].numpy().tolist()
        bit_32_input, bit_32_output = bit_32[i, :-1].numpy().tolist(), bit_32[i, 1:].numpy().tolist()
        bit_33_input, bit_33_output = bit_33[i, :-1].numpy().tolist(), bit_33[i, 1:].numpy().tolist()
        bit_34_input, bit_34_output = bit_34[i, :-1].numpy().tolist(), bit_34[i, 1:].numpy().tolist()
        bit_35_input, bit_35_output = bit_35[i, :-1].numpy().tolist(), bit_35[i, 1:].numpy().tolist()
        bit_36_input, bit_36_output = bit_36[i, :-1].numpy().tolist(), bit_36[i, 1:].numpy().tolist()
        bit_37_input, bit_37_output = bit_37[i, :-1].numpy().tolist(), bit_37[i, 1:].numpy().tolist()
        bit_38_input, bit_38_output = bit_38[i, :-1].numpy().tolist(), bit_38[i, 1:].numpy().tolist()
        bit_39_input, bit_39_output = bit_39[i, :-1].numpy().tolist(), bit_39[i, 1:].numpy().tolist()
        bit_40_input, bit_40_output = bit_40[i, :-1].numpy().tolist(), bit_40[i, 1:].numpy().tolist()
        bit_41_input, bit_41_output = bit_41[i, :-1].numpy().tolist(), bit_41[i, 1:].numpy().tolist()
        bit_42_input, bit_42_output = bit_42[i, :-1].numpy().tolist(), bit_42[i, 1:].numpy().tolist()
        bit_43_input, bit_43_output = bit_43[i, :-1].numpy().tolist(), bit_43[i, 1:].numpy().tolist()
        bit_44_input, bit_44_output = bit_44[i, :-1].numpy().tolist(), bit_44[i, 1:].numpy().tolist()
        bit_45_input, bit_45_output = bit_45[i, :-1].numpy().tolist(), bit_45[i, 1:].numpy().tolist()
        bit_46_input, bit_46_output = bit_46[i, :-1].numpy().tolist(), bit_46[i, 1:].numpy().tolist()
        bit_47_input, bit_47_output = bit_47[i, :-1].numpy().tolist(), bit_47[i, 1:].numpy().tolist()
        bit_48_input, bit_48_output = bit_48[i, :-1].numpy().tolist(), bit_48[i, 1:].numpy().tolist()
        bit_49_input, bit_49_output = bit_49[i, :-1].numpy().tolist(), bit_49[i, 1:].numpy().tolist()
        bit_50_input, bit_50_output = bit_50[i, :-1].numpy().tolist(), bit_50[i, 1:].numpy().tolist()
        bit_51_input, bit_51_output = bit_51[i, :-1].numpy().tolist(), bit_51[i, 1:].numpy().tolist()
        bit_52_input, bit_52_output = bit_52[i, :-1].numpy().tolist(), bit_52[i, 1:].numpy().tolist()
        bit_53_input, bit_53_output = bit_53[i, :-1].numpy().tolist(), bit_53[i, 1:].numpy().tolist()
        bit_54_input, bit_54_output = bit_54[i, :-1].numpy().tolist(), bit_54[i, 1:].numpy().tolist()
        bit_55_input, bit_55_output = bit_55[i, :-1].numpy().tolist(), bit_55[i, 1:].numpy().tolist()
        bit_56_input, bit_56_output = bit_56[i, :-1].numpy().tolist(), bit_56[i, 1:].numpy().tolist()
        bit_57_input, bit_57_output = bit_57[i, :-1].numpy().tolist(), bit_57[i, 1:].numpy().tolist()
        bit_58_input, bit_58_output = bit_58[i, :-1].numpy().tolist(), bit_58[i, 1:].numpy().tolist()
        bit_59_input, bit_59_output = bit_59[i, :-1].numpy().tolist(), bit_59[i, 1:].numpy().tolist()
        bit_60_input, bit_60_output = bit_60[i, :-1].numpy().tolist(), bit_60[i, 1:].numpy().tolist()
        bit_61_input, bit_61_output = bit_61[i, :-1].numpy().tolist(), bit_61[i, 1:].numpy().tolist()
        bit_62_input, bit_62_output = bit_62[i, :-1].numpy().tolist(), bit_62[i, 1:].numpy().tolist()
        bit_63_input, bit_63_output = bit_63[i, :-1].numpy().tolist(), bit_63[i, 1:].numpy().tolist()

        inputs = tf.data.Dataset.from_tensor_slices((bit_0_input,bit_1_input,bit_2_input,bit_3_input,bit_4_input,bit_5_input,bit_6_input,
                                                     bit_7_input,bit_8_input,bit_9_input,bit_10_input,bit_11_input,bit_12_input,bit_13_input,
                                                     bit_14_input,bit_15_input,bit_16_input,bit_17_input,bit_18_input,bit_19_input,bit_20_input,
                                                     bit_21_input,bit_22_input,bit_23_input,bit_24_input,bit_25_input,bit_26_input,bit_27_input,
                                                     bit_28_input,bit_29_input,bit_30_input,bit_31_input,bit_32_input,bit_33_input,bit_34_input,
                                                     bit_35_input,bit_36_input,bit_37_input,bit_38_input,bit_39_input,bit_40_input,bit_41_input,
                                                     bit_42_input,bit_43_input,bit_44_input,bit_45_input,bit_46_input,bit_47_input,bit_48_input,
                                                     bit_49_input,bit_50_input,bit_51_input,bit_52_input,bit_53_input,bit_54_input,bit_55_input,
                                                     bit_56_input,bit_57_input,bit_58_input,bit_59_input,bit_60_input,bit_61_input,bit_62_input,
                                                     bit_63_input))
        outputs= tf.data.Dataset.from_tensor_slices((bit_0_output,bit_1_output,bit_2_output,bit_3_output,bit_4_output,bit_5_output,bit_6_output,
                                                     bit_7_output,bit_8_output,bit_9_output,bit_10_output,bit_11_output,bit_12_output,bit_13_output,
                                                     bit_14_output,bit_15_output,bit_16_output,bit_17_output,bit_18_output,bit_19_output,bit_20_output,
                                                     bit_21_output,bit_22_output,bit_23_output,bit_24_output,bit_25_output,bit_26_output,bit_27_output,
                                                     bit_28_output,bit_29_output,bit_30_output,bit_31_output,bit_32_output,bit_33_output,bit_34_output,
                                                     bit_35_output,bit_36_output,bit_37_output,bit_38_output,bit_39_output,bit_40_output,bit_41_output,
                                                     bit_42_output,bit_43_output,bit_44_output,bit_45_output,bit_46_output,bit_47_output,bit_48_output,
                                                     bit_49_output,bit_50_output,bit_51_output,bit_52_output,bit_53_output,bit_54_output,bit_55_output,
                                                     bit_56_output,bit_57_output,bit_58_output,bit_59_output,bit_60_output,bit_61_output,bit_62_output,
                                                     bit_63_output))
        inputs = inputs.batch(batch_size, drop_remainder=True)
        outputs = outputs.batch(batch_size, drop_remainder=True)
        # train_data=tf.data.Dataset.zip((inputs,outputs)).batch(batch_size,drop_remainder=True) #if we want to do prediction using the whole data
        def loss(labels, logits):
            return tf.keras.losses.binary_crossentropy(labels, logits)

        packet_loss = []
        bit_loss = 0
        counter = 0
        for inp, oup in zip(inputs.as_numpy_iterator(), outputs.as_numpy_iterator()):
            prediction = model.predict(inp)  # this gives prediction for all the 64 bits
            for pred, true_oup in zip(prediction, oup):  # bit level predictions
                counter = counter + 1
                pre = pred.reshape(1, -1)
                bit_loss = bit_loss + loss(true_oup, pre)
            bit_loss = bit_loss.numpy().tolist()
            packet_loss.append(bit_loss[0] / counter)
            bit_loss = 0
            counter = 0
        whole_sequence_loss.append(sum(packet_loss) / len(packet_loss))
    return whole_sequence_loss

