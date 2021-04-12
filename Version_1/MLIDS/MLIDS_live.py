import frames_fetcher
import bits_extractor
import read_temp_frame
import testing_dataset_creator_with_time

all_ids =['04b1', '00a1', '0430', '02a0', '0130', '0329', '0545', '0370', '05f0', '0316', '0002', '0260', '02b0',
          '05a2', '0440','0140', '0131', '0350', '018f', '0153', '05a0', '0690', '04f0', '043f', '00a0', '02c0', '01f1']  # available arbitration IDs in the dataset

all_ids_length = {'04b1': 50, '00a1': 10, '0430': 50, '02a0': 100, '0130': 100, '0329': 100, '0545': 100, '0370': 100,
                  '05f0': 20, '0316': 100, '0002': 100, '0260': 100, '02b0': 100, '05a2': 1, '0440': 100, '0140': 100,
                  '0131': 100, '0350': 101, '018f': 100, '0153': 100, '05a0': 1, '0690': 10, '04f0': 50, '043f': 100,
                  '00a0': 10, '02c0': 101, '01f1': 50}              # arbitration ID and its corresponing frequency in a second
freq = list(all_ids_length.values())


duration=1               # detection duration
frames_fetcher.fetch(duration)     # fetch packets from the terminal that is playing the can dataset
batch_size=1                       # we need to change the batch size to 1 inorder to make predictions
#===================================================
file = open("temp_file.txt", "r")    #read frames collected by frame fetcher
Arb_id_0 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[0])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_0)

arb_id_0 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../Arb_id_0/training_checkpoints') # this folder is not available unless all arbitration IDs are trained first

#===================================================
file = open("temp_file.txt", "r")
Arb_id_2 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[2])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_2)

arb_id_2 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../Arb_id_2/training_checkpoints')
#===================================================

file = open("temp_file.txt", "r")
Arb_id_3 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[0])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_0)

arb_id_3 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../Arb_id_0/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_4 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[4])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_4)

arb_id_4 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../Arb_id_4/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_5 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[5])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_5)

arb_id_5 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../Arb_id_5/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_6 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[6])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_6)

arb_id_6 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../Arb_id_6/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_7 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[7])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_7)

arb_id_7 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../Arb_id_7/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_9 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[9])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_4)

arb_id_9 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../Arb_id_9/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_10 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[10])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_10)

arb_id_10 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                                batch_size=batch_size, model_dir='../Arb_id_10/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_11 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[11])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_11)

arb_id_11 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                                batch_size=batch_size, model_dir='../Arb_id_11/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_12 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[12])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_12)

arb_id_12 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                                batch_size=batch_size, model_dir='../Arb_id_12/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_14 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[14])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_14)

arb_id_14 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                                batch_size=batch_size, model_dir='../Arb_id_14/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_15 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[15])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_15)

arb_id_15 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                                batch_size=batch_size, model_dir='../Arb_id_15/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_16 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[16])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_16)

arb_id_16 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                                batch_size=batch_size, model_dir='../Arb_id_16/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_17 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[17])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_17)

arb_id_17 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                                batch_size=batch_size, model_dir='../Arb_id_17/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_18 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[18])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_18)

arb_id_18 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                                batch_size=batch_size, model_dir='../Arb_id_18/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_19 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[19])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_19)

arb_id_19 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                                batch_size=batch_size, model_dir='../Arb_id_19/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_22 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[22])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_22)

arb_id_22 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                                batch_size=batch_size, model_dir='../Arb_id_22/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_23 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[23])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_23)

arb_id_23 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                                batch_size=batch_size, model_dir='../Arb_id_23/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_25 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[25])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_25)

arb_id_25 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                                batch_size=batch_size, model_dir='../Arb_id_25/training_checkpoints')
#===================================================
file = open("temp_file.txt", "r")
Arb_id_26 = read_temp_frame.prepare_dataset(file, arbitration_id=all_ids[26])  # total, sequencelength, 64  # 700,68,64
file.close()
bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_26)

arb_id_26 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                                batch_size=batch_size, model_dir='../Arb_id_26/training_checkpoints')
#===================================================

anomaly_signal=(freq[0]*arb_id_0+freq[2]*arb_id_2+freq[3]*arb_id_3+freq[4]*arb_id_4+freq[5]*arb_id_5+freq[6]*arb_id_6+freq[7]*arb_id_7+freq[11]*arb_id_11+
     freq[14]*arb_id_14+freq[15]*arb_id_15+freq[16]*arb_id_16+freq[17]*arb_id_17+freq[18]*arb_id_18+freq[19]*arb_id_19+freq[22]*arb_id_22+freq[23]*arb_id_23+
     freq[25]*arb_id_25+freq[26]*arb_id_26)\
    /(freq[0]+freq[2]+freq[3]+freq[4]+freq[5]+freq[6]+freq[7]+freq[9]+freq[11]+freq[12]+freq[14]+freq[15]+freq[16]+freq[17]+freq[18]+freq[19]+
      freq[22]+freq[23]+freq[25]+freq[26])

if anomaly_signal > 1.12:
    print("Anomaly")
