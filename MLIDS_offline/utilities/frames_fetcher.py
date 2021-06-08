import time
import string

def read_file_tolist(att_ty, att_frq):  # attack type and attack frequency
    file_name = open('../datasets/prepared_attacks/' + att_ty + '_' + str(att_frq) + '.txt', "r")
    data_lines = []
    for two in file_name:
        data_lines.append(two)
    return data_lines


def fetch(begn_time,attack_type,attack_freq, duration, data_lines):
    current_window_data = open('temp_'+str(attack_type)+'_'+str(attack_freq)+'.txt', 'w')
    for rows in data_lines:
        arr_time = float(rows[1:18])
        time_stamp=rows[1:18]
        arb_id = rows[25:33].lower()
        data = rows[34:50].lower()
        row_data = time_stamp + ',' + arb_id + ',' + data + '\n'
        if begn_time <= arr_time <= begn_time + duration:
            current_window_data.write(row_data)
        elif arr_time > begn_time + duration:
            begn_time = arr_time
            break
    current_window_data.close()
    return begn_time


if __name__ == "__main__":
    attack_type = 'insertion_attack'
    attack_freq = 0.01
    file = open('../datasets/prepared_attacks/' + attack_type + '_' + str(attack_freq) + '.txt', "r")
    for line in file:
        initial_time = float(line[1:18])
        break
    file.close()
    all_packets = read_file_tolist(attack_type, attack_freq)
    while True:
        a = fetch(initial_time, duration=1, data_lines=all_packets)
        print(initial_time, a)
        initial_time = a
        break
