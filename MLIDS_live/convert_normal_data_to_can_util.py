#In order to play the datasets from 'https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset' in python-can we need to change it to can-util format
def can_util_format(file):
    output_file=open('../datasets/normal_data_can_util.log', 'w')
    for line in file:
        arr_time=line[11:28]
        arb_id = line[41:44]
        data = line[65:88].replace(' ','')
        can_util_frame='('+arr_time+')' + ' ' + 'vcan0' + ' ' + arb_id + '#' + data +'\n'
        output_file.write(can_util_frame)
    output_file.close()

input_file=can_util_format(open('../datasets/normal_run_data.txt', 'r'))
