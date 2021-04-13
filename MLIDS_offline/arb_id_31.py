import LSTM_IDS

arb_index = 31
testing_duration = 1000
attack_type = 'insertion_attack'
attack_freq = 0.01
det_window = 1
LSTM_IDS.test_each_ID(arb_index, testing_duration, attack_type, attack_freq, det_window)
