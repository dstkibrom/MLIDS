# indexs=[0,1,2,3,8,9,10,11,13,16,27,28,29,30,31,32,33,34,35,37,39]
#
# testing_duration=2000
# for index in indexs:
#     file=open('arb_id_'+str(index)+'.py','w')
#     file.write('import LSTM_IDS\n\n')
#     file.write('arb_index = '+str(index)+'\n')
#     file.write('testing_duration = '+str(testing_duration)+'\n')
#     file.write('attack_type= [\'drop_attack\',\'fuzzy_attack\',\'insertion_attack\']\n')
#     file.write('attack_freq=[0.05,0.04,0.03,0.02,0.01]\n')
#     file.write('det_window = 1\n')
#     file.write('for att_ty in attack_type:\n')
#     file.write('\tfor att_fr in attack_freq:\n')
#     file.write('\t\tLSTM_IDS.test_each_ID(arb_index, testing_duration, att_ty, att_fr, det_window)\n')
#     file.close()

indexs=[0,1,2,3,8,9,10,11,13,16,27,28,29,30,31,32,33,34,35,37,39]

for index in indexs:
    file=open('new_shellscripts/arb_id_'+str(index)+'.sh','w')
    file.write('#!/bin/bash\n')
    file.write('#$ -S /bin/bash\n')
    file.write('#$ -cwd\n')
    file.write('#$ -V\n')
    file.write('export PATH=\"/work/araya-kd/anaconda3/bin:$PATH\"\n')
    file.write('source activate tf_gpu_cuda8\n')
    file.write('cd /work/araya-kd/MLIDS/MLIDS_offline\n')
    file.write('python3 arb_id_'+str(index)+'.py')
    file.close()

