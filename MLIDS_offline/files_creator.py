indexs=[0,1,2,3,8,9,10,11,13,16,27,28,29,30,31,32,33,34,35,37,39]

testing_duration=1000
attack_freq=0.01
for index in indexs:
    file=open('arb_id_'+str(index)+'.py','w')
    file.write('import LSTM_IDS\n\n')
    file.write('arb_index = '+str(index)+'\n')
    file.write('testing_duration = '+str(testing_duration)+'\n')
    file.write('attack_type = \'insertion_attack\'\n')
    file.write('attack_freq = '+str(attack_freq)+'\n')
    file.write('det_window = 1\n')
    file.write('LSTM_IDS.test_each_ID(arb_index, testing_duration, attack_type, attack_freq, det_window)\n')
    file.close()

# indexs=[0,1,2,3,8,9,10,11,13,16,27,28,29,30,31,32,33,34,35,37,39]
#
# testing_duration=1
# attack_freq=0.01
# for index in indexs:
#     file=open('shellscripts/arb_id_'+str(index)+'.sh','w')
#     file.write('#!/bin/bash\n')
#     file.write('#$ -S /bin/bash\n')
#     file.write('#$ -cwd\n')
#     file.write('#$ -V\n')
#     file.write('export PATH=\"/work/araya-kd/anaconda3/bin:$PATH\"\n')
#     file.write('source activate tf_gpu_cuda8\n')
#     file.write('cd /work/araya-kd/MLIDS/MLIDS_offline\n')
#     file.write('python3 arb_id_'+str(index)+'.py')
#     file.close()
