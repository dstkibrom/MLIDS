import tensorflow as tf
import numpy as np

all_ids_length = {'0CF00400': 50, '0CF00300': 20, '18FEF100': 10, '1CFF6F00': 10, '18ECFF00': 1, '18FF8800': 1, '18FF8400': 2,
                  '18FEE500': 4, '18F00029': 10, '18FEF200': 10, '18FF7F00': 10, '1CFF7100': 10, '18EBFF00': 5, '18FF8200': 10,
                  '18FF8600': 2, '18FEDC00': 4, '1CFF7700': 10, '18FF8900': 1, '18FEDF00': 4, '18FEE900': 3, '18FF8700': 2,
                  '18FEE700': 3, '1CFEB300': 3, '18FEC100': 1, '18FEEE00': 1, '18ECFF29': 1, '18EBFF29': 3, '0C000027': 100,
                  '0C000F27': 20, '18FEF111': 10, '0CF00203': 99, '0CF00327': 100, '18FF8327': 100, '0C002927': 20,
                  '18FF5027': 10, '18F00503': 10, '18FF5127': 1, '18FEED11': 10, '18FEE617': 1, '1CFFAA27': 10, '18EC0027': 0, '18EB0027': 0}


def ready_for_training(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12,
                       bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25,
                       bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, bit_37, bit_38,
                       bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52,
                       bit_53, bit_54, bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63, batch_size, duration,
                       memory_size, arbitration_id):

    sequence_length=int(all_ids_length[arbitration_id] * duration)      # in 1 second how many messages of that specific ID
    max_element_size=int(len(bit_0)/(sequence_length+1))
    max_element_size=max_element_size*(sequence_length+1)          # maximum number of elements that can be created

    bit_0 = bit_0[:max_element_size];bit_1 = bit_1[:max_element_size];bit_2 = bit_2[:max_element_size];bit_3 = bit_3[:max_element_size]
    bit_4 = bit_4[:max_element_size];bit_5 = bit_5[:max_element_size];bit_6 = bit_6[:max_element_size];bit_7 = bit_7[:max_element_size]
    bit_8 = bit_8[:max_element_size];bit_9 = bit_9[:max_element_size];bit_10 = bit_10[:max_element_size];bit_11 = bit_11[:max_element_size]
    bit_12 = bit_12[:max_element_size];bit_13 = bit_13[:max_element_size];bit_14 = bit_14[:max_element_size];bit_15 = bit_15[:max_element_size]
    bit_16 = bit_16[:max_element_size];bit_17 = bit_17[:max_element_size];bit_18 = bit_18[:max_element_size];bit_19 = bit_19[:max_element_size]
    bit_20 = bit_20[:max_element_size];bit_21 = bit_21[:max_element_size];bit_22 = bit_22[:max_element_size];bit_23 = bit_23[:max_element_size]
    bit_24 = bit_24[:max_element_size];bit_25 = bit_25[:max_element_size];bit_26 = bit_26[:max_element_size];bit_27 = bit_27[:max_element_size]
    bit_28 = bit_28[:max_element_size];bit_29 = bit_29[:max_element_size];bit_30 = bit_30[:max_element_size];bit_31 = bit_31[:max_element_size]
    bit_32 = bit_32[:max_element_size];bit_33 = bit_33[:max_element_size];bit_34 = bit_34[:max_element_size];bit_35 = bit_35[:max_element_size]
    bit_36 = bit_36[:max_element_size];bit_37 = bit_37[:max_element_size];bit_38 = bit_38[:max_element_size];bit_39 = bit_39[:max_element_size]
    bit_40 = bit_40[:max_element_size];bit_41 = bit_41[:max_element_size];bit_42 = bit_42[:max_element_size];bit_43 = bit_43[:max_element_size]
    bit_44 = bit_44[:max_element_size];bit_45 = bit_45[:max_element_size];bit_46 = bit_46[:max_element_size];bit_47 = bit_47[:max_element_size]
    bit_48 = bit_48[:max_element_size];bit_49 = bit_49[:max_element_size];bit_50 = bit_50[:max_element_size];bit_51 = bit_51[:max_element_size]
    bit_52 = bit_52[:max_element_size];bit_53 = bit_53[:max_element_size];bit_54 = bit_54[:max_element_size];bit_55 = bit_55[:max_element_size]
    bit_56 = bit_56[:max_element_size];bit_57 = bit_57[:max_element_size];bit_58 = bit_58[:max_element_size];bit_59 = bit_59[:max_element_size]
    bit_60 = bit_60[:max_element_size];bit_61 = bit_61[:max_element_size];bit_62 = bit_62[:max_element_size];bit_63 = bit_63[:max_element_size]

    bit_0 = np.array(bit_0).reshape(-1, sequence_length + 1);bit_1 = np.array(bit_1).reshape(-1, sequence_length + 1)
    bit_2 = np.array(bit_2).reshape(-1, sequence_length + 1);bit_3 = np.array(bit_3).reshape(-1, sequence_length + 1)
    bit_4 = np.array(bit_4).reshape(-1, sequence_length + 1);bit_5 = np.array(bit_5).reshape(-1, sequence_length + 1)
    bit_6 = np.array(bit_6).reshape(-1, sequence_length + 1);bit_7 = np.array(bit_7).reshape(-1, sequence_length + 1)
    bit_8 = np.array(bit_8).reshape(-1, sequence_length + 1);bit_9 = np.array(bit_9).reshape(-1, sequence_length + 1)
    bit_10 = np.array(bit_10).reshape(-1, sequence_length + 1);bit_11 = np.array(bit_11).reshape(-1, sequence_length + 1)
    bit_12 = np.array(bit_12).reshape(-1, sequence_length + 1);bit_13 = np.array(bit_13).reshape(-1, sequence_length + 1)
    bit_14 = np.array(bit_14).reshape(-1, sequence_length + 1);bit_15 = np.array(bit_15).reshape(-1, sequence_length + 1)
    bit_16 = np.array(bit_16).reshape(-1, sequence_length + 1);bit_17 = np.array(bit_17).reshape(-1, sequence_length + 1)
    bit_18 = np.array(bit_18).reshape(-1, sequence_length + 1);bit_19 = np.array(bit_19).reshape(-1, sequence_length + 1)
    bit_20 = np.array(bit_20).reshape(-1, sequence_length + 1);bit_21 = np.array(bit_21).reshape(-1, sequence_length + 1)
    bit_22 = np.array(bit_22).reshape(-1, sequence_length + 1);bit_23 = np.array(bit_23).reshape(-1, sequence_length + 1)
    bit_24 = np.array(bit_24).reshape(-1, sequence_length + 1);bit_25 = np.array(bit_25).reshape(-1, sequence_length + 1)
    bit_26 = np.array(bit_26).reshape(-1, sequence_length + 1);bit_27 = np.array(bit_27).reshape(-1, sequence_length + 1)
    bit_28 = np.array(bit_28).reshape(-1, sequence_length + 1);bit_29 = np.array(bit_29).reshape(-1, sequence_length + 1)
    bit_30 = np.array(bit_30).reshape(-1, sequence_length + 1);bit_31 = np.array(bit_31).reshape(-1, sequence_length + 1)
    bit_32 = np.array(bit_32).reshape(-1, sequence_length + 1);bit_33 = np.array(bit_33).reshape(-1, sequence_length + 1)
    bit_34 = np.array(bit_34).reshape(-1, sequence_length + 1);bit_35 = np.array(bit_35).reshape(-1, sequence_length + 1)
    bit_36 = np.array(bit_36).reshape(-1, sequence_length + 1);bit_37 = np.array(bit_37).reshape(-1, sequence_length + 1)
    bit_38 = np.array(bit_38).reshape(-1, sequence_length + 1);bit_39 = np.array(bit_39).reshape(-1, sequence_length + 1)
    bit_40 = np.array(bit_40).reshape(-1, sequence_length + 1);bit_41 = np.array(bit_41).reshape(-1, sequence_length + 1)
    bit_42 = np.array(bit_42).reshape(-1, sequence_length + 1);bit_43 = np.array(bit_43).reshape(-1, sequence_length + 1)
    bit_44 = np.array(bit_44).reshape(-1, sequence_length + 1);bit_45 = np.array(bit_45).reshape(-1, sequence_length + 1)
    bit_46 = np.array(bit_46).reshape(-1, sequence_length + 1);bit_47 = np.array(bit_47).reshape(-1, sequence_length + 1)
    bit_48 = np.array(bit_48).reshape(-1, sequence_length + 1);bit_49 = np.array(bit_49).reshape(-1, sequence_length + 1)
    bit_50 = np.array(bit_50).reshape(-1, sequence_length + 1);bit_51 = np.array(bit_51).reshape(-1, sequence_length + 1)
    bit_52 = np.array(bit_52).reshape(-1, sequence_length + 1);bit_53 = np.array(bit_53).reshape(-1, sequence_length + 1)
    bit_54 = np.array(bit_54).reshape(-1, sequence_length + 1);bit_55 = np.array(bit_55).reshape(-1, sequence_length + 1)
    bit_56 = np.array(bit_56).reshape(-1, sequence_length + 1);bit_57 = np.array(bit_57).reshape(-1, sequence_length + 1)
    bit_58 = np.array(bit_58).reshape(-1, sequence_length + 1);bit_59 = np.array(bit_59).reshape(-1, sequence_length + 1)
    bit_60 = np.array(bit_60).reshape(-1, sequence_length + 1);bit_61 = np.array(bit_61).reshape(-1, sequence_length + 1)
    bit_62 = np.array(bit_62).reshape(-1, sequence_length + 1);bit_63 = np.array(bit_63).reshape(-1, sequence_length + 1)

    bit_0_input, bit_0_output = bit_0[:, :-1], bit_0[:, 1:];bit_1_input, bit_1_output = bit_1[:, :-1], bit_1[:, 1:]
    bit_2_input, bit_2_output = bit_2[:, :-1], bit_2[:, 1:];bit_3_input, bit_3_output = bit_3[:, :-1], bit_3[:, 1:]
    bit_4_input, bit_4_output = bit_4[:, :-1], bit_4[:, 1:];bit_5_input, bit_5_output = bit_5[:, :-1], bit_5[:, 1:]
    bit_6_input, bit_6_output = bit_6[:, :-1], bit_6[:, 1:];bit_7_input, bit_7_output = bit_7[:, :-1], bit_7[:, 1:]
    bit_8_input, bit_8_output = bit_8[:, :-1], bit_8[:, 1:];bit_9_input, bit_9_output = bit_9[:, :-1], bit_9[:, 1:]
    bit_10_input, bit_10_output = bit_10[:, :-1], bit_10[:, 1:];bit_11_input, bit_11_output = bit_11[:, :-1], bit_11[:, 1:]
    bit_12_input, bit_12_output = bit_12[:, :-1], bit_12[:, 1:];bit_13_input, bit_13_output = bit_13[:, :-1], bit_13[:, 1:]
    bit_14_input, bit_14_output = bit_14[:, :-1], bit_14[:, 1:];bit_15_input, bit_15_output = bit_15[:, :-1], bit_15[:, 1:]
    bit_16_input, bit_16_output = bit_16[:, :-1], bit_16[:, 1:];bit_17_input, bit_17_output = bit_17[:, :-1], bit_17[:, 1:]
    bit_18_input, bit_18_output = bit_18[:, :-1], bit_18[:, 1:];bit_19_input, bit_19_output = bit_19[:, :-1], bit_19[:, 1:]
    bit_20_input, bit_20_output = bit_20[:, :-1], bit_20[:, 1:];bit_21_input, bit_21_output = bit_21[:, :-1], bit_21[:, 1:]
    bit_22_input, bit_22_output = bit_22[:, :-1], bit_22[:, 1:];bit_23_input, bit_23_output = bit_23[:, :-1], bit_23[:, 1:]
    bit_24_input, bit_24_output = bit_24[:, :-1], bit_24[:, 1:];bit_25_input, bit_25_output = bit_25[:, :-1], bit_25[:, 1:]
    bit_26_input, bit_26_output = bit_26[:, :-1], bit_26[:, 1:];bit_27_input, bit_27_output = bit_27[:, :-1], bit_27[:, 1:]
    bit_28_input, bit_28_output = bit_28[:, :-1], bit_28[:, 1:];bit_29_input, bit_29_output = bit_29[:, :-1], bit_29[:, 1:]
    bit_30_input, bit_30_output = bit_30[:, :-1], bit_30[:, 1:];bit_31_input, bit_31_output = bit_31[:, :-1], bit_31[:, 1:]
    bit_32_input, bit_32_output = bit_32[:, :-1], bit_32[:, 1:];bit_33_input, bit_33_output = bit_33[:, :-1], bit_33[:, 1:]
    bit_34_input, bit_34_output = bit_34[:, :-1], bit_34[:, 1:];bit_35_input, bit_35_output = bit_35[:, :-1], bit_35[:, 1:]
    bit_36_input, bit_36_output = bit_36[:, :-1], bit_36[:, 1:];bit_37_input, bit_37_output = bit_37[:, :-1], bit_37[:, 1:]
    bit_38_input, bit_38_output = bit_38[:, :-1], bit_38[:, 1:];bit_39_input, bit_39_output = bit_39[:, :-1], bit_39[:, 1:]
    bit_40_input, bit_40_output = bit_40[:, :-1], bit_40[:, 1:];bit_41_input, bit_41_output = bit_41[:, :-1], bit_41[:, 1:]
    bit_42_input, bit_42_output = bit_42[:, :-1], bit_42[:, 1:];bit_43_input, bit_43_output = bit_43[:, :-1], bit_43[:, 1:]
    bit_44_input, bit_44_output = bit_44[:, :-1], bit_44[:, 1:];bit_45_input, bit_45_output = bit_45[:, :-1], bit_45[:, 1:]
    bit_46_input, bit_46_output = bit_46[:, :-1], bit_46[:, 1:];bit_47_input, bit_47_output = bit_47[:, :-1], bit_47[:, 1:]
    bit_48_input, bit_48_output = bit_48[:, :-1], bit_48[:, 1:];bit_49_input, bit_49_output = bit_49[:, :-1], bit_49[:, 1:]
    bit_50_input, bit_50_output = bit_50[:, :-1], bit_50[:, 1:];bit_51_input, bit_51_output = bit_51[:, :-1], bit_51[:, 1:]
    bit_52_input, bit_52_output = bit_52[:, :-1], bit_52[:, 1:];bit_53_input, bit_53_output = bit_53[:, :-1], bit_53[:, 1:]
    bit_54_input, bit_54_output = bit_54[:, :-1], bit_54[:, 1:];bit_55_input, bit_55_output = bit_55[:, :-1], bit_55[:, 1:]
    bit_56_input, bit_56_output = bit_56[:, :-1], bit_56[:, 1:];bit_57_input, bit_57_output = bit_57[:, :-1], bit_57[:, 1:]
    bit_58_input, bit_58_output = bit_58[:, :-1], bit_58[:, 1:];bit_59_input, bit_59_output = bit_59[:, :-1], bit_59[:, 1:]
    bit_60_input, bit_60_output = bit_60[:, :-1], bit_60[:, 1:];bit_61_input, bit_61_output = bit_61[:, :-1], bit_61[:, 1:]
    bit_62_input, bit_62_output = bit_62[:, :-1], bit_62[:, 1:];bit_63_input, bit_63_output = bit_63[:, :-1], bit_63[:, 1:]

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
    train_dataset = tf.data.Dataset.zip((inputs, outputs)).batch(batch_size, drop_remainder=True).shuffle(memory_size)

    return train_dataset


def id_sequence_ready_for_training(id_sequence, batch_size, duration, memory_size):
    tr_id_sequence = tf.data.Dataset.from_tensor_slices(id_sequence)
    tr_id_sequence = tr_id_sequence.batch(332 * duration, drop_remainder=True)

    def split_input_target(chunk):
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        return input_ids, target_ids

    # split the the data set to input and output
    tr_id_sequence = tr_id_sequence.map(split_input_target)

    # Shuffle and batch
    tr_id_sequence = tr_id_sequence.shuffle(memory_size).batch(batch_size, drop_remainder=True)

    return tr_id_sequence
