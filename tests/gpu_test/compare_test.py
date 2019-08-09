import os
import time

begin_1060 = time.time()
os.system("/usr/local/bin/python3.6 /home/noio0925/Desktop/RL-repo/NerveNet/tests/gpu_test/lstm_test.py --gpu_index=1")
end_1060 = time.time()

begin_2070 = time.time()
os.system("/usr/local/bin/python3.6 /home/noio0925/Desktop/RL-repo/NerveNet/tests/gpu_test/lstm_test.py --gpu_index=0")
end_2070 = time.time()

print("1060 takes: ", end_1060 - begin_1060)
print("2070 takes: ", end_2070 - begin_2070)
