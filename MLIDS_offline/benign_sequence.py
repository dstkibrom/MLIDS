def main():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from utilities import frames_fetcher, temp_frame_extractor, bits_extractor,predictor
    import concurrent.futures


    all_ids = ['0CF00400', '0CF00300', '18FEF100', '1CFF6F00', '18ECFF00', '18FF8800', '18FF8400',
               '18FEE500', '18F00029', '18FEF200', '18FF7F00', '1CFF7100', '18EBFF00', '18FF8200',
               '18FF8600', '18FEDC00', '1CFF7700', '18FF8900', '18FEDF00', '18FEE900', '18FF8700',
               '18FEE700', '1CFEB300', '18FEC100', '18FEEE00', '18ECFF29', '18EBFF29', '0C000027',
               '0C000F27', '18FEF111', '0CF00203', '0CF00327', '18FF8327', '0C002927', '18FF5027',
               '18F00503', '18FF5127', '18FEED11', '18FEE617', '1CFFAA27', '18EC0027', '18EB0027']

    all_ids_length = {'0CF00400': 50, '0CF00300': 20, '18FEF100': 10, '1CFF6F00': 10, '18ECFF00': 1, '18FF8800': 1,
                      '18FF8400': 2,
                      '18FEE500': 4, '18F00029': 10, '18FEF200': 10, '18FF7F00': 10, '1CFF7100': 10, '18EBFF00': 5,
                      '18FF8200': 10,
                      '18FF8600': 2, '18FEDC00': 4, '1CFF7700': 10, '18FF8900': 1, '18FEDF00': 4, '18FEE900': 3,
                      '18FF8700': 2,
                      '18FEE700': 3, '1CFEB300': 3, '18FEC100': 1, '18FEEE00': 1, '18ECFF29': 1, '18EBFF29': 3,
                      '0C000027': 100,
                      '0C000F27': 20, '18FEF111': 10, '0CF00203': 99, '0CF00327': 100, '18FF8327': 100, '0C002927': 20,
                      '18FF5027': 10, '18F00503': 10, '18FF5127': 1, '18FEED11': 10, '18FEE617': 1, '1CFFAA27': 10,
                      '18EC0027': 0, '18EB0027': 0}
    arb_frequency = list(all_ids_length.values())
    arb_id_indexs = [0, 1, 2, 3, 8, 9, 10, 11, 13, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39]

    batch_size = 1  # we need to change the batch size to 1 inorder to make predictions


    def get_id_prediction(arb_index):
        file = "temp_file.txt"  # read frames collected by frame fetcher which collects frames from a file in the datasets

        Arb_id = temp_frame_extractor.prepare_dataset(file, arbitration_id=all_ids[
            arb_index])  # returns two consecutive packets of the same arbitration IDs

        bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7, bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14, bit_15, bit_16, bit_17, bit_18, \
        bit_19, bit_20, bit_21, bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28, bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35, bit_36, \
        bit_37, bit_38, bit_39, bit_40, bit_41, bit_42, bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49, bit_50, bit_51, bit_52, bit_53, bit_54, \
        bit_55, bit_56, bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63 = bits_extractor.extract_all_bits(Arb_id)

        arb_id = predictor.ready_for_testing(bit_0, bit_1, bit_2, bit_3, bit_4, bit_5, bit_6, bit_7,
                                             bit_8, bit_9, bit_10, bit_11, bit_12, bit_13, bit_14,
                                             bit_15, bit_16, bit_17, bit_18, bit_19, bit_20, bit_21,
                                             bit_22, bit_23, bit_24, bit_25, bit_26, bit_27, bit_28,
                                             bit_29, bit_30, bit_31, bit_32, bit_33, bit_34, bit_35,
                                             bit_36, bit_37, bit_38, bit_39, bit_40, bit_41, bit_42,
                                             bit_43, bit_44, bit_45, bit_46, bit_47, bit_48, bit_49,
                                             bit_50, bit_51, bit_52, bit_53, bit_54, bit_55, bit_56,
                                             bit_57, bit_58, bit_59, bit_60, bit_61, bit_62, bit_63,
                                             batch_size=batch_size,
                                             model_dir='../trained_models/' + str(
                                                 arb_index) + '/training_checkpoints')  # this folder is not available unless all arbitration IDs are trained first
        return arb_id[0]

    duration=1     # detection duration
    attack_type = 'benign_data'
    attack_freq = 0
    breaking_duration=1

    file = open('../datasets/prepared_attacks/' + attack_type + '_' + str(attack_freq) + '.txt', "r")
    for line in file:
        initial_time = float(line[1:18])
        break

    file.close()
    all_packets = frames_fetcher.read_file_tolist(attack_type, attack_freq)
    counter=0           # a counter for breaking out of the while loop
    results_file=open('../new_results/'+attack_type+'_'+str(attack_freq)+'.txt','w')
    signal_values=[]   # collection of the results
    while True:
        counter=counter+1
        next_initial_time = frames_fetcher.fetch(initial_time, duration=1, data_lines=all_packets)
        initial_time = next_initial_time + 5.0   # the 5.0 is so as for the system to skip every 5 seconds for faster testing

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(get_id_prediction, arb_id_indexs)

        anomaly_signal = 0
        for freq, res in zip(arb_frequency, results):
            anomaly_signal = anomaly_signal + freq * res
        anomaly_signal = anomaly_signal / sum(arb_frequency)
        signal_values.append(anomaly_signal)

        if counter >= breaking_duration:
            break

    results_file.write(str(signal_values))
    results_file.close()


if __name__ == '__main__':
    main()