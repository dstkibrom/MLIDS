import tensorflow as tf
all_ids = ['04b1', '00a1', '0430', '02a0', '0130', '0329', '0545', '0370', '05f0', '0316', '0002', '0260', '02b0',
          '05a2', '0440','0140', '0131', '0350', '018f', '0153', '05a0', '0690', '04f0', '043f', '00a0', '02c0', '01f1']


def prepare_dataset_anomaly(file, det_duration, arbitration_id):
    counter = 0
    arb_id_data = []
    sarb_id_data=[]    # arb ID collector in a second
    initial_time=0
    for line in file:
        initial_time = float(line[:17])   # read the first arrival time in the file
        break

    for line in file:
        arr_time = float(line[:17])
        arb_id = line[18:22]
        data = line[25:-3].replace(',','')   #-3, 'comman,R or T and '\n'

        if arb_id == arbitration_id and arr_time - initial_time < det_duration:
            while len(data)<16:   #This is for fuzzy atack, coz we never know the size of DLC
                data=data+'00'
            data = list(str(format(int(data[:2], 16), '08b')) + str(format(int(data[2:4], 16), '08b')) + str(
                format(int(data[4:6], 16), '08b'))
                        + str(format(int(data[6:8], 16), '08b')) + str(format(int(data[8:10], 16), '08b')) + str(
                format(int(data[10:12], 16), '08b'))
                        + str(format(int(data[12:14], 16), '08b')) + str(format(int(data[14:16], 16), '08b')))
            data = [int(i) for i in data]  # convert the string values to int
            sarb_id_data.append(data)
            counter = counter + 1  # for counting the numbers of data sets

        elif arr_time - initial_time >= det_duration:
            arb_id_data.append(sarb_id_data)
            sarb_id_data = []
            initial_time = arr_time
            if arb_id == arbitration_id:
                while len(data) < 16:
                    data = data + '00'
                data = list(str(format(int(data[:2], 16), '08b')) + str(format(int(data[2:4], 16), '08b')) + str(
                    format(int(data[4:6], 16), '08b'))
                            + str(format(int(data[6:8], 16), '08b')) + str(format(int(data[8:10], 16), '08b')) + str(
                    format(int(data[10:12], 16), '08b'))
                            + str(format(int(data[12:14], 16), '08b')) + str(format(int(data[14:16], 16), '08b')))
                data = [int(i) for i in data]  # convert the string values to int
                sarb_id_data.append(data)
    return tf.ragged.constant(arb_id_data)

def prepare_dataset_benign(file,det_duration, arbitration_id):
    counter = 0
    arb_id_data = []
    sarb_id_data=[]    # arb ID collector in a second
    initial_time=0
    for line in file:
        initial_time = float(line[11:28])   # read the first arrival time in the file
        break
    for line in file:
        arr_time = float(line[11:27])
        arb_id = line[40:44]
        data = line[65:88].replace(' ', '')

        if arb_id == arbitration_id and arr_time - initial_time < det_duration:
            while len(data)<16:   #This is for fuzzy atack, coz we never know the size of DLC
                data=data+'00'
            data = list(str(format(int(data[:2], 16), '08b')) + str(format(int(data[2:4], 16), '08b')) + str(
                format(int(data[4:6], 16), '08b'))
                        + str(format(int(data[6:8], 16), '08b')) + str(format(int(data[8:10], 16), '08b')) + str(
                format(int(data[10:12], 16), '08b'))
                        + str(format(int(data[12:14], 16), '08b')) + str(format(int(data[14:16], 16), '08b')))
            data = [int(i) for i in data]  # convert the string values to int
            sarb_id_data.append(data)
            counter = counter + 1  # for counting the numbers of data sets

        elif arr_time - initial_time >= det_duration:
            arb_id_data.append(sarb_id_data)
            sarb_id_data = []
            initial_time = arr_time
            if arb_id == arbitration_id:
                while len(data) < 16:
                    data = data + '00'
                data = list(str(format(int(data[:2], 16), '08b')) + str(format(int(data[2:4], 16), '08b')) + str(
                    format(int(data[4:6], 16), '08b'))
                            + str(format(int(data[6:8], 16), '08b')) + str(format(int(data[8:10], 16), '08b')) + str(
                    format(int(data[10:12], 16), '08b'))
                            + str(format(int(data[12:14], 16), '08b')) + str(format(int(data[14:16], 16), '08b')))
                data = [int(i) for i in data]  # convert the string values to int
                sarb_id_data.append(data)
    return tf.ragged.constant(arb_id_data)