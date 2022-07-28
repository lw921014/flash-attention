import argparse
import numpy as np
import logging
import logging.handlers

logger = logging.getLogger("demo-infer")
logger.setLevel(logging.DEBUG)

rf_handler = logging.StreamHandler()
rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

logger.addHandler(rf_handler)

def check_gmem_tile(file_name):
    passed = True
    with open(file_name, 'r') as fp:
        id = 0
        for line in fp:
            id = id + 1
            line = line.strip()
            # print(line.split(","))
            real_row = int(line.split(",")[9].strip().split("=")[1].strip())
            real_col = int(line.split(",")[10].strip().split("=")[1].strip())
            value = int(float(line.split(",")[11].strip().split("=")[1].strip()))
            preds = int(float(line.split(",")[12].strip().split("=")[1].strip()))
            value = abs(value) 

            expect = real_row * 64 + real_col
            if preds == 1 and expect != value:
                if abs(expect - value) > 1:
                    print(f"line id = {id}")
                    print(f"context = {line}")
                    print(f"row{real_row}, col{real_col}: expect {real_row * 64 + real_col}, but {value}")
                    print("====================================")
                    passed = False

    if passed:
        print("passed")
    else:
        print("failed")

def check_p():
    m = 100
    n = 32

    q = np.array(np.arange(m * n), dtype = np.float16).reshape((m, n))
    k = np.array(np.arange(m * n), dtype = np.float16).reshape((m, n)).transpose(1, 0)
    p = np.matmul(q, k)

    print(q)
    print(k)
    print(p)
    np.savetxt("q.data", q, fmt='%d', delimiter=',')
    np.savetxt("k.data", k, fmt='%d', delimiter=',')
    np.savetxt("p.data", p, fmt='%d', delimiter=',')

# def gemm(A, B, C, relu = True):
#     ret = np.matmul(A, B) + C
#     return np.maximum(ret, 0) if relu else ret

if __name__ == "__main__":
    check_gmem_tile("Gmem_tile_mask_load_after.log")
    # check_p()