import argparse
import numpy as np
import logging
import logging.handlers
from operator import itemgetter, attrgetter
from check_tool import is_same_matrix, get_array_from_file 

logger = logging.getLogger("demo-infer")
logger.setLevel(logging.DEBUG)

rf_handler = logging.StreamHandler()
rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

logger.addHandler(rf_handler)

def get_orrd(l, threadIdx, i_m, id):
    warp_id = int(threadIdx / 32) 
    lane_id = threadIdx % 32
    quad_id = int(lane_id / 4)
    t_id = lane_id % 4
    
    # ii = int(id / 4)
    jj = id % 4
    ii = i_m
    ni = int(id / 4)
    print(f"warp_id = {warp_id}, lane_id = {lane_id}, quad_id = {quad_id}, t_id = {t_id}, id = {id}, "
          f"mi = {0}, ni = {ni}, ii = {ii}, jj = {jj}, ")

    row = l * 16 + ii * 8 + quad_id
    col = ni * 64 + warp_id * 16 + t_id * 2 #+ jj % 2 * 2 + int(jj / 2) * 8 + jj % 2
    if jj < 2:
        col += jj
    else:
        col += jj + 8 - 2

    return [int(row), int(col)]

def check_gmem_tile(file_name):
    passed = True
    with open(file_name, 'r') as fp:
        id = 0
        all_coord = []
        for line in fp:
            id = id + 1
            line = line.strip()
            if line.startswith("after_mask") and line.find("inf") != -1:
                print(f"find inf in {line}")
                l = int(get_ele_from_line(line, "l"))
                loop_step_idx = int(get_ele_from_line(line, "loop_step_idx"))
                threadIdx = int(get_ele_from_line(line, "threadIdx.x"))
                i_m = int(get_ele_from_line(line, "i_m"))

                all_id = get_inf_id_from_line(line)
                print(f"l = {l}, loop_step_idx = {loop_step_idx}, threadIdx = {threadIdx}, i_m = {i_m}")
                for id in all_id:
                    tmp = get_orrd(l, threadIdx, i_m, id)
                    all_coord.append(tmp)
                    print(tmp)
                    # print(all_coord)
                # if threadIdx == 44:
                #     exit()

        all_coord = sorted(all_coord, key=itemgetter(0,1))
        for i in range(len(all_coord)):
            if i == 0:
                continue
            if all_coord[i][0] ==  all_coord[i-1][0] and \
                all_coord[i][1] ==  all_coord[i-1][1]:
                print (f"repeated pos = {all_coord[i]}")
                exit(1)

        print (all_coord)

        row = 0
        col = 0
        for coord in all_coord:
            row = max(row, coord[0])
            col = max(col, coord[1])
        row = row +1
        col = col +1
        mask = np.zeros((row, col))
        for coord in all_coord:
            mask[coord[0]][coord[1]] = 999
        
        np.savetxt("gen_mask.data", mask, fmt='%d', delimiter=',')

        mask_ref = get_array_from_file("attn_mask_op.data", "test").reshape((6,64,64))[0,:,:]
        mask_ref = np.squeeze(mask_ref)

        # test 999
        # gr -1
        def check(test, gr, reverse = False):
            test_value = 999 if not reverse else -1
            gr_value = -1 if not reverse else 999
            
            test_mask = test if not reverse else gr
            gr_mask = gr if not reverse else test

            mask_ok = True
            count = 0
            error_list = []
            for i in range(gr_mask.shape[0]):
                for j in range(gr_mask.shape[1]):
                    if gr_mask[i][j] == gr_value and (i < 49 and j < 49):
                        if i >= test_mask.shape[0] or j >= test_mask.shape[1]:
                            print(f"i = {i}, j = {j} not find, mask failed")
                            mask_ok = False
                            count += 1
                            error_list.append([i, j])
                        elif test_mask[i][j] != test_value:
                            print(f"i = {i}, j = {j}, mask failed")
                            mask_ok = False
                            count += 1
                            error_list.append([i, j])
                        # else:
                        #     print("ok")
            if mask_ok:
                print("passed")
            else:
                print(f"failed, error count = {count}, pos list is {error_list}")

        check(mask, mask_ref)
        check(mask, mask_ref, True)

    print("%s" % ("passed" if is_true else "failed"))
    print("\n================================================\n")
    return is_true

def check_kernel():
    q_data = get_array_from_file("q_op.data", "test").reshape((49,1,32))
    k_data = get_array_from_file("k_op.data", "test").reshape((49,1,32))
    v_data = get_array_from_file("v_op.data", "test").reshape((49,1,32))
    o_data = get_array_from_file("o_op.data", "test").reshape((49,1,32))

    q_data = np.squeeze(q_data)
    k_data = np.squeeze(k_data)
    v_data = np.squeeze(v_data)
    o_data = np.squeeze(o_data)

    def sofmax(logits):
        e_x = np.exp(logits - np.max(logits))
        probs = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return probs

    p = np.matmul(q_data, k_data.transpose(1,0)) / np.sqrt(32)
    s = sofmax(p)
    o_cmp = np.matmul(s, v_data)
    
    print(o_cmp)
    print(o_data)

    is_same_matrix(o_cmp, o_data, abs_eps = 0.01)

if __name__ == "__main__":
    # check_gmem_tile("after_mask.log")
    check_kernel()