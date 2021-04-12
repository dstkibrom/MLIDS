all_ids = ['0CF00400', '0CF00300', '18FEF100', '1CFF6F00', '18ECFF00', '18FF8800', '18FF8400',
           '18FEE500', '18F00029', '18FEF200', '18FF7F00', '1CFF7100', '18EBFF00', '18FF8200',
           '18FF8600', '18FEDC00', '1CFF7700', '18FF8900', '18FEDF00', '18FEE900', '18FF8700',
           '18FEE700', '1CFEB300', '18FEC100', '18FEEE00', '18ECFF29', '18EBFF29', '0C000027',
           '0C000F27', '18FEF111', '0CF00203', '0CF00327', '18FF8327', '0C002927', '18FF5027',
           '18F00503', '18FF5127', '18FEED11', '18FEE617', '1CFFAA27', '18EC0027', '18EB0027']


def prepare_dataset(file, arbitration_id):
    counter = 0
    # collect data for the length of duration and append it to the above variables
    sid_data_sequence=[]
    for line in file:
        counter = counter + 1  # for counting the numbers of data sets , this should be here for synchronization
        arb_id = line[18:22]
        data = line[23:-1]

        if arb_id == arbitration_id:
            while len(data) < 16:
                data=data+'00'
            data = list(str(format(int(data[:2], 16), '08b')) + str(format(int(data[2:4], 16), '08b')) + str(
                format(int(data[4:6], 16), '08b'))
                        + str(format(int(data[6:8], 16), '08b')) + str(format(int(data[8:10], 16), '08b')) + str(
                format(int(data[10:12], 16), '08b'))
                        + str(format(int(data[12:14], 16), '08b')) + str(format(int(data[14:16], 16), '08b')))
            data = [int(i) for i in data]  # convert the string values to int
            sid_data_sequence.append(data)
    return sid_data_sequence[-2:]
# sid_data_sequence[-2:] if only the last packet is needed

if __name__ == "__main__":
    print("In file")
    print(prepare_dataset(open('temp_file.txt','r'),all_ids[0]))