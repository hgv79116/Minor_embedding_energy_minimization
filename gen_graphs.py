import sys
from random import randint

def gen_w(w_low, w_high): 
    return randint(w_low, w_high) * (1 if randint(0, 1) else -1)

if __name__ == "__main__": 
    test_num = int(sys.argv[1])
    n_low = int(sys.argv[2])
    n_high = int(sys.argv[3])
    w_low = int(sys.argv[4])
    w_high = int(sys.argv[5])
    bias_scalar = int(sys.argv[6])

    print(test_num)

    for test in range(test_num): 
        n = randint(n_low, n_high)
        print(n)
        assert w_low >= 0 and w_high >= 0, "value_lim must not be negative"
        for i in range(n): 
            w = gen_w(w_low, w_high) * bias_scalar
            print(w, end = " " if i + 1 < n else "\n")
        
        for i in range(n): 
            for j in range(i + 1, n): 
                print(gen_w(w_low, w_high))

