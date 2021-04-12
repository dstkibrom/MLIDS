all_ids = ['0CF00400', '0CF00300', '18FEF100', '1CFF6F00', '18ECFF00', '18FF8800', '18FF8400',
           '18FEE500', '18F00029', '18FEF200', '18FF7F00', '1CFF7100', '18EBFF00', '18FF8200',
           '18FF8600', '18FEDC00', '1CFF7700', '18FF8900', '18FEDF00', '18FEE900', '18FF8700',
           '18FEE700', '1CFEB300', '18FEC100', '18FEEE00', '18ECFF29', '18EBFF29', '0C000027',
           '0C000F27', '18FEF111', '0CF00203', '0CF00327', '18FF8327', '0C002927', '18FF5027',
           '18F00503', '18FF5127', '18FEED11', '18FEE617', '1CFFAA27', '18EC0027', '18EB0027']


def prepare_dataset(file, min_index, max_index, arbitration_id):
    counter = 0
    # collect data for the length of duration and append it to the above variables
    sid_sequence = []
    sid=[]

    for line in file:
        counter = counter + 1  # for counting the numbers of data sets , this should be here for synchronization
        arb_id = line[25:33]
        data = line[34:-1]
        #         data=str(format(int(data, 16),'064b'))
        data = list(str(format(int(data[:2], 16), '08b')) + str(format(int(data[2:4], 16), '08b')) + str(
            format(int(data[4:6], 16), '08b')) \
                    + str(format(int(data[6:8], 16), '08b')) + str(format(int(data[8:10], 16), '08b')) + str(
            format(int(data[10:12], 16), '08b')) \
                    + str(format(int(data[12:14], 16), '08b')) + str(format(int(data[14:], 16), '08b')))
        data = [int(i) for i in data]  # convert the string values to int
        sid_sequence.append(all_ids.index(arb_id))

        if arb_id == arbitration_id:
            sid.append(data)

        if counter > max_index:
            break

    return sid_sequence[min_index:max_index], sid[min_index:max_index]
