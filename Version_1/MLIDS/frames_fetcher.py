import can
import time


def fetch(duration):
    bus = can.interface.Bus(bustype="socketcan", channel="vcan0", bitrate=500000)
    current_window_data = open('temp_file.txt', 'w')
    start = time.time()
    duration = duration
    for msg in bus:
        frame = str(msg)
        time_stamp = frame[11:28]
        arb_id = frame[36:44]
        data = frame[76:99].replace(' ', '')
        row_data = time_stamp + ',' + arb_id + ',' + data + '\n'
        current_window_data.write(row_data)
        if time.time() - start > duration:
            current_window_data.close()
            break

    return None

if __name__ == "__main__":
    print("in file")
    fetch(1)

