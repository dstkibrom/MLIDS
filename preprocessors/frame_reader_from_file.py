all_ids=['04b1', '00a1', '0430', '02a0', '0130', '0329', '0545', '0370', '05f0', '0316', '0002', '0260', '02b0', '05a2', '0440',
         '0140', '0131', '0350', '018f', '0153', '05a0', '0690', '04f0', '043f', '00a0', '02c0', '01f1']


def prepare_dataset(file, arbitration_id):
    counter = 0
    # collect data for the length of duration and append it to the above variables
    sid_data_sequence=[]

    for line in file:
        counter = counter + 1  # for counting the numbers of data sets , this should be here for synchronization
        arb_id = line[40:44]
        data = line[65:88].replace(' ','')
        if arb_id == arbitration_id:
            data = list(str(format(int(data[:2], 16), '08b')) + str(format(int(data[2:4], 16), '08b')) + str(
                format(int(data[4:6], 16), '08b'))
                        + str(format(int(data[6:8], 16), '08b')) + str(format(int(data[8:10], 16), '08b')) + str(
                format(int(data[10:12], 16), '08b'))
                        + str(format(int(data[12:14], 16), '08b')) + str(format(int(data[14:16], 16), '08b')))
            data = [int(i) for i in data]  # convert the string values to int
            sid_data_sequence.append(data)
    return sid_data_sequence