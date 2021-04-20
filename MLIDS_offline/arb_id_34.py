import LSTM_IDS

arb_index = 34
testing_duration = 2000
attack_type= ['drop_attack','fuzzy_attack','insertion_attack']
attack_freq=[0.05,0.04,0.03,0.02,0.01]
det_window = 1
for att_ty in attack_type:
	for att_fr in attack_freq:
		LSTM_IDS.test_each_ID(arb_index, testing_duration, att_ty, att_fr, det_window)
