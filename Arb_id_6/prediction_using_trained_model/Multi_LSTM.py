import logging
import preprocessors.frame_reader_with_time as frame_reader_with_time
import preprocessors.bits_extractor_with_time as bits_extractor_with_time
import testing_dataset_creator_with_time

logging.getLogger('tensorflow').disabled = True

all_ids =['04b1', '00a1', '0430', '02a0', '0130', '0329', '0545', '0370', '05f0', '0316', '0002', '0260', '02b0',
          '05a2', '0440','0140', '0131', '0350', '018f', '0153', '05a0', '0690', '04f0', '043f', '00a0', '02c0', '01f1']


batch_size=1
arb_id=all_ids[6]
file = open("../../datasets/Fuzzy_dataset.csv", "r")
test_data = frame_reader_with_time.prepare_dataset_anomaly(file, det_duration=1, arbitration_id=arb_id)  # total, sequencelength, 64  # 700,68,64

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor_with_time.extract_all_bits(test_data)

input_test_data = testing_dataset_creator_with_time.ready_for_training(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
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
                                                    batch_size=batch_size)
print(input_test_data)

file=open("arbID_6",'a')
file.write("\n\nAnomaly results\n\n"+str(input_test_data))
file.close()
print("+++++++++++++++++++++++++++++++++++")
file = open("../../datasets/test_data.txt", "r")
test_data = frame_reader_with_time.prepare_dataset_benign(file, det_duration=1, arbitration_id=arb_id)  # total, sequencelength, 64  # 700,68,64

bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor_with_time.extract_all_bits(test_data)


input_test_data = testing_dataset_creator_with_time.ready_for_training(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
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
                                                    batch_size=batch_size)
file=open("arbID_0",'a')
file.write("\n\nBenign results\n\n"+str(input_test_data))
file.close()
print(input_test_data)