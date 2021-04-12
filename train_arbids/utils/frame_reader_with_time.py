import tensorflow as tf
all_ids = ['0CF00400', '0CF00300', '18FEF100', '1CFF6F00', '18ECFF00', '18FF8800', '18FF8400',
           '18FEE500', '18F00029', '18FEF200', '18FF7F00', '1CFF7100', '18EBFF00', '18FF8200',
           '18FF8600', '18FEDC00', '1CFF7700', '18FF8900', '18FEDF00', '18FEE900', '18FF8700',
           '18FEE700', '1CFEB300', '18FEC100', '18FEEE00', '18ECFF29', '18EBFF29', '0C000027',
           '0C000F27', '18FEF111', '0CF00203', '0CF00327', '18FF8327', '0C002927', '18FF5027',
           '18F00503', '18FF5127', '18FEED11', '18FEE617', '1CFFAA27', '18EC0027', '18EB0027']


def prepare_dataset(file, det_duration, arbitration_id, dur_seconds):
    counter = 0
    arb_id_data = []
    sarb_id_data=[]
    initial_time=0
    for line in file:
        initial_time = float(line[1:18])   # read the first arrival time in the file
        break
    time_breaker=initial_time
    for line in file:
        arr_time = float(line[1:18])
        arb_id = line[25:33]
        data = line[34:-1]
        #         data=str(format(int(data, 16),'064b'))
        data = list(str(format(int(data[:2], 16), '08b')) + str(format(int(data[2:4], 16), '08b')) + str(
            format(int(data[4:6], 16), '08b')) \
                    + str(format(int(data[6:8], 16), '08b')) + str(format(int(data[8:10], 16), '08b')) + str(
            format(int(data[10:12], 16), '08b')) \
                    + str(format(int(data[12:14], 16), '08b')) + str(format(int(data[14:16], 16), '08b')))
        data = [int(i) for i in data]  # convert the string values to int

        if arb_id == arbitration_id and arr_time - initial_time < det_duration:
            counter = counter + 1  # for counting the numbers of data sets
            sarb_id_data.append(data)
        elif arb_id == arbitration_id and arr_time -initial_time >= det_duration:
            arb_id_data.append(sarb_id_data)
            sarb_id_data = []
            sarb_id_data.append(data)
            initial_time = arr_time
        if arr_time > time_breaker + dur_seconds:
            break
    return tf.ragged.constant(arb_id_data)

