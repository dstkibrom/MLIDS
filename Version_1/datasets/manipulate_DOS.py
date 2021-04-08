import tensorflow as tf
import random
all_ids = ['04b1', '00a1', '0430', '02a0', '0130', '0329', '0545', '0370', '05f0', '0316', '0002', '0260', '02b0',
          '05a2', '0440','0140', '0131', '0350', '018f', '0153', '05a0', '0690', '04f0', '043f', '00a0', '02c0', '01f1']


file=open('datasets/DoS_dataset.csv','r')
DoS=open('datasets/DoS_dataset_manipulated.txt','a')
for line in file:
    arr_time = line[:18]
    arb_id = line[18:22]
    data=line[22:]
    if arb_id == '0000':
        arb_id= '0002'
    frame =  arr_time+arb_id+data
    DoS.write(frame)
DoS.close()
