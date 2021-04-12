import logging
# from utils import frame_reader_with_time, testing_dataset_creator_with_time, bits_extractor_with_time
import frames_fetcher,temp_frame_extractor,bits_extractor

all_ids = ['0CF00400', '0CF00300', '18FEF100', '1CFF6F00', '18ECFF00', '18FF8800', '18FF8400',
           '18FEE500', '18F00029', '18FEF200', '18FF7F00', '1CFF7100', '18EBFF00', '18FF8200',
           '18FF8600', '18FEDC00', '1CFF7700', '18FF8900', '18FEDF00', '18FEE900', '18FF8700',
           '18FEE700', '1CFEB300', '18FEC100', '18FEEE00', '18ECFF29', '18EBFF29', '0C000027',
           '0C000F27', '18FEF111', '0CF00203', '0CF00327', '18FF8327', '0C002927', '18FF5027',
           '18F00503', '18FF5127', '18FEED11', '18FEE617', '1CFFAA27', '18EC0027', '18EB0027']

all_ids_length = {'0CF00400': 50, '0CF00300': 20, '18FEF100': 10, '1CFF6F00': 10, '18ECFF00': 1, '18FF8800': 1, '18FF8400': 2,
                  '18FEE500': 4, '18F00029': 10, '18FEF200': 10, '18FF7F00': 10, '1CFF7100': 10, '18EBFF00': 5, '18FF8200': 10,
                  '18FF8600': 2, '18FEDC00': 4, '1CFF7700': 10, '18FF8900': 1, '18FEDF00': 4, '18FEE900': 3, '18FF8700': 2,
                  '18FEE700': 3, '1CFEB300': 3, '18FEC100': 1, '18FEEE00': 1, '18ECFF29': 1, '18EBFF29': 3, '0C000027': 100,
                  '0C000F27': 20, '18FEF111': 10, '0CF00203': 99, '0CF00327': 100, '18FF8327': 100, '0C002927': 20,
                  '18FF5027': 10, '18F00503': 10, '18FF5127': 1, '18FEED11': 10, '18FEE617': 1, '1CFFAA27': 10, '18EC0027': 0, '18EB0027': 0}
freq = list(all_ids_length.values())


duration=1                         # detection duration
frames_fetcher.fetch(duration)     # fetch packets from the terminal that is playing the can dataset
batch_size=1                       # we need to change the batch size to 1 inorder to make predictions

#===================================================
file = "temp_file.txt"     #read frames collected by frame fetcher


Arb_id_0 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[0])  # total, sequencelength, 64  # 700,68,64
print(Arb_id_0)
exit(0)
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_0)
#
# arb_id_0 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                batch_size=batch_size, model_dir='../Arb_id_0/training_checkpoints') # this folder is not available unless all arbitration IDs are trained first
#
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_2 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[2])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_2)
#
# arb_id_2 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                batch_size=batch_size, model_dir='../Arb_id_2/training_checkpoints')
# #===================================================
#
# file = open("temp_file.txt", "r")
# Arb_id_3 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[0])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_0)
#
# arb_id_3 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                batch_size=batch_size, model_dir='../Arb_id_0/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_4 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[4])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_4)
#
# arb_id_4 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                batch_size=batch_size, model_dir='../Arb_id_4/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_5 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[5])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_5)
#
# arb_id_5 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                batch_size=batch_size, model_dir='../Arb_id_5/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_6 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[6])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_6)
#
# arb_id_6 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                batch_size=batch_size, model_dir='../Arb_id_6/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_7 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[7])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_7)
#
# arb_id_7 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                batch_size=batch_size, model_dir='../Arb_id_7/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_9 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[9])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_4)
#
# arb_id_9 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                batch_size=batch_size, model_dir='../Arb_id_9/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_10 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[10])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_10)
#
# arb_id_10 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                 bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                 bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                 bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                 bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                 bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                 bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                 bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                 bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                 batch_size=batch_size, model_dir='../Arb_id_10/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_11 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[11])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_11)
#
# arb_id_11 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                 bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                 bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                 bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                 bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                 bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                 bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                 bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                 bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                 batch_size=batch_size, model_dir='../Arb_id_11/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_12 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[12])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_12)
#
# arb_id_12 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                 bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                 bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                 bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                 bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                 bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                 bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                 bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                 bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                 batch_size=batch_size, model_dir='../Arb_id_12/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_14 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[14])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_14)
#
# arb_id_14 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                 bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                 bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                 bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                 bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                 bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                 bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                 bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                 bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                 batch_size=batch_size, model_dir='../Arb_id_14/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_15 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[15])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_15)
#
# arb_id_15 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                 bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                 bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                 bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                 bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                 bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                 bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                 bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                 bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                 batch_size=batch_size, model_dir='../Arb_id_15/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_16 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[16])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_16)
#
# arb_id_16 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                 bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                 bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                 bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                 bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                 bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                 bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                 bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                 bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                 batch_size=batch_size, model_dir='../Arb_id_16/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_17 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[17])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_17)
#
# arb_id_17 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                 bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                 bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                 bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                 bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                 bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                 bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                 bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                 bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                 batch_size=batch_size, model_dir='../Arb_id_17/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_18 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[18])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_18)
#
# arb_id_18 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                 bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                 bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                 bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                 bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                 bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                 bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                 bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                 bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                 batch_size=batch_size, model_dir='../Arb_id_18/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_19 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[19])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_19)
#
# arb_id_19 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                 bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                 bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                 bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                 bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                 bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                 bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                 bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                 bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                 batch_size=batch_size, model_dir='../Arb_id_19/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_22 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[22])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_22)
#
# arb_id_22 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                 bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                 bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                 bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                 bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                 bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                 bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                 bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                 bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                 batch_size=batch_size, model_dir='../Arb_id_22/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_23 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[23])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_23)
#
# arb_id_23 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                 bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                 bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                 bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                 bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                 bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                 bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                 bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                 bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                 batch_size=batch_size, model_dir='../Arb_id_23/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_25 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[25])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_25)
#
# arb_id_25 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                 bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                 bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                 bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                 bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                 bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                 bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                 bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                 bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                 batch_size=batch_size, model_dir='../Arb_id_25/training_checkpoints')
# #===================================================
# file = open("temp_file.txt", "r")
# Arb_id_26 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[26])  # total, sequencelength, 64  # 700,68,64
# file.close()
# bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
# bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
# bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
# bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_26)
#
# arb_id_26 = testing_dataset_creator_with_time.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
#                                                                 bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
#                                                                 bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
#                                                                 bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
#                                                                 bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
#                                                                 bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
#                                                                 bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
#                                                                 bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
#                                                                 bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
#                                                                 batch_size=batch_size, model_dir='../Arb_id_26/training_checkpoints')
# #===================================================
#
# anomaly_signal=(freq[0]*arb_id_0+freq[2]*arb_id_2+freq[3]*arb_id_3+freq[4]*arb_id_4+freq[5]*arb_id_5+freq[6]*arb_id_6+freq[7]*arb_id_7+freq[11]*arb_id_11+
#      freq[14]*arb_id_14+freq[15]*arb_id_15+freq[16]*arb_id_16+freq[17]*arb_id_17+freq[18]*arb_id_18+freq[19]*arb_id_19+freq[22]*arb_id_22+freq[23]*arb_id_23+
#      freq[25]*arb_id_25+freq[26]*arb_id_26)\
#     /(freq[0]+freq[2]+freq[3]+freq[4]+freq[5]+freq[6]+freq[7]+freq[9]+freq[11]+freq[12]+freq[14]+freq[15]+freq[16]+freq[17]+freq[18]+freq[19]+
#       freq[22]+freq[23]+freq[25]+freq[26])
#
# if anomaly_signal > 1.12:
#     print("Anomaly")
