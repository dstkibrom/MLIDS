import logging
# from utils import frame_reader_with_time, testing_dataset_creator_with_time, bits_extractor_with_time
import frames_fetcher,temp_frame_extractor,bits_extractor
import predictor
import numpy as np

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


Arb_id_0 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[0])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_0)

arb_id_0 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/0/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_0)
print("========================================================")
Arb_id_1 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[1])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_1)

arb_id_1 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/1/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first

print(arb_id_1)
print("========================================================")
Arb_id_2 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[2])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_2)

arb_id_2 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/2/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first

print(arb_id_2)
print("========================================================")
Arb_id_3 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[3])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_3)

arb_id_3 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/3/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_3)
print("========================================================")

Arb_id_8 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[8])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_8)

arb_id_8 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/8/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first

print(arb_id_8)
print("========================================================")
Arb_id_9 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[9])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_9)

arb_id_9 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/9/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_9)
print("========================================================")

Arb_id_10 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[10])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_10)

arb_id_10 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/10/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_10)
print("========================================================")

Arb_id_11 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[11])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_11)

arb_id_11 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/11/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_11)
print("========================================================")

Arb_id_13 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[13])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_13)

arb_id_13 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/13/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_13)
print("========================================================")

Arb_id_16 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[16])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_16)

arb_id_16 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/16/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_16)
print("========================================================")

Arb_id_27 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[27])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_27)

arb_id_27 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/27/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_27)
print("========================================================")

Arb_id_28 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[28])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_28)

arb_id_28 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/28/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_28)
print("========================================================")

Arb_id_29 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[29])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_29)

arb_id_29 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/29/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_29)
print("========================================================")

Arb_id_30 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[30])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_30)

arb_id_30 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/30/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_30)
print("========================================================")

Arb_id_31 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[31])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_31)

arb_id_31 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/31/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_31)
print("========================================================")

Arb_id_32 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[32])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_32)

arb_id_32 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/32/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_32)
print("========================================================")

Arb_id_33 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[33])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_33)

arb_id_33 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/33/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_33)
print("========================================================")

Arb_id_34 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[34])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_34)

arb_id_34 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/34/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_34)
print("========================================================")


Arb_id_35 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[35])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_35)

arb_id_35 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/35/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_35)
print("========================================================")

Arb_id_37 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[37])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_37)

arb_id_37 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/37/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_37)
print("========================================================")

Arb_id_39 = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[39])   #  returns two consecutive packets of the same arbitration IDs

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id_39)

arb_id_39 = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                                               bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                                               bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                                               bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                                               bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                                               bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                                               bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                                               bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                                               bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                                               batch_size=batch_size, model_dir='../trained_models/39/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
print(arb_id_39)
print("========================================================")


print(arb_id_0,arb_id_1,arb_id_2,arb_id_3,arb_id_8,arb_id_9,arb_id_10,arb_id_11,arb_id_13,arb_id_16,arb_id_27,
      arb_id_28,arb_id_29,arb_id_30,arb_id_31,arb_id_32,arb_id_33,arb_id_34,arb_id_35,arb_id_37,arb_id_39)


anomaly_signal=(freq[0] * arb_id_0[0] +
                freq[2] * arb_id_2[0] +
                freq[3] * arb_id_3[0] +
                freq[8] * arb_id_8[0] +
                freq[9] * arb_id_9[0] +
                freq[10] * arb_id_10[0] +
                freq[11] * arb_id_11[0] +
                freq[13] * arb_id_13[0] +
                freq[16] * arb_id_16[0] +
                freq[27] * arb_id_27[0] +
                freq[28] * arb_id_28[0] +
                freq[29] * arb_id_29[0] +
                freq[30] * arb_id_30[0] +
                freq[31] * arb_id_31[0] +
                freq[32] * arb_id_32[0] +
                freq[33] * arb_id_33[0] +
                freq[34] * arb_id_34[0] +
                freq[35] * arb_id_35[0] +
                freq[37] * arb_id_37[0] +
                freq[39] * arb_id_39[0] )/(freq[0] + freq[1] +
                                        freq[2] + freq[3] +
                                        freq[8] + freq[9] +
                                        freq[10] + freq[11] +
                                        freq[13] + freq[16] +
                                        freq[27] + freq[28] +
                                        freq[29] + freq[30] +
                                        freq[31] + freq[32] +
                                        freq[33] + freq[34] +
                                        freq[35] + freq[37] +
                                        freq[39])


if anomaly_signal > 1.12:
    print("Anomaly")
else:
    print("Benign")
print(anomaly_signal)
