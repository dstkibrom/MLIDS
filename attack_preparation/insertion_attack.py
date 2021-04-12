import random
all_ids = ['0CF00400', '0CF00300', '18FEF100', '1CFF6F00', '18ECFF00', '18FF8800', '18FF8400',
           '18FEE500', '18F00029', '18FEF200', '18FF7F00', '1CFF7100', '18EBFF00', '18FF8200',
           '18FF8600', '18FEDC00', '1CFF7700', '18FF8900', '18FEDF00', '18FEE900', '18FF8700',
           '18FEE700', '1CFEB300', '18FEC100', '18FEEE00', '18ECFF29', '18EBFF29', '0C000027',
           '0C000F27', '18FEF111', '0CF00203', '0CF00327', '18FF8327', '0C002927', '18FF5027',
           '18F00503', '18FF5127', '18FEED11', '18FEE617', '1CFFAA27', '18EC0027', '18EB0027']

# 332 packets in a second
# def prepare_dataset(file, inser_freq):

file = open("test_data_attack_prep.txt", "r")
packets=[]
insertion_attack = open("insertion_attack.txt", 'a')
for line in file:
    packets.append(line)
file.close()

inser_freq=0.01
insert_every=int(332*inser_freq)
counter=0
for i in range(len(packets)):
    counter=counter+1
    insertion_attack.write(packets[i])
    if counter == insert_every:
        f_half= packets[i][:19]                   # this is to select the time when the attack will happen
        s_half=packets[random.randint(0, len(packets))][19:-1] + '\t Attack\n'
        insertion_attack.write(f_half+s_half)  # select random packet and insert it at this specific location
        counter=0
insertion_attack.close()


