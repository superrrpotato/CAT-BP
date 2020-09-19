import numpy as np
def spike_distance_v4(str1, str2, mem1):
    # str1 is the original spike sequence, mem1 is the original membrane potential
    # str2 is the new spike sequence
    delta = 0.8
    tau_m = 5
    cost = [0] * (len(str1) + 1)
    paired_flag = False
    for i in range(len(str1)):
        if i == 0:
            if str1[0] != str2[0]:
                if str1[0] == 1:
                    cost[1] = min(mem1[0]-1,1)
                    mem1[i+1:i+2]+=delta
                else:
                    cost[1] = min(1-mem1[0],1)
                    mem1[i+1:i+2]-=mem1[i]*(1-1/tau_m)
        else:
            if mem1[i]>1 and str1[i]==0:
                str1[i]=1
                mem1[i+1:i+2] -= (mem1[i]-delta)*(1-1/tau_m)
            elif mem1[i]<1 and str1[i]==1:
                str1[i]=0
                mem1[i+1:i+2] += mem1[i]*(1-1/tau_m)
            if str1[i] == str2[i]:
                cost[i+1] = cost[i]
                continue
            else:
                if paired_flag == False and str1[i-1] == str2[i] and str1[i] == str2[i-1]:
                    if str1[i] == 1:
                        cost[i+1] = cost[i-1] + (max(min(mem1[i]-1,1),0) + max(min(1-mem1[i-1],1),0))/2
                        mem1[i+1:i+2] += delta
                    else:
                        cost[i+1] = cost[i-1] + (max(min(1-mem1[i],1),0) + max(min(mem1[i-1]-1,1),0))/2
                        mem1[i+1:i+2]-=mem1[i]*(1-1/tau_m)
                    paired_flag = True
                else:
                    if str1[i] == 1:
                        cost[i+1] = cost[i] + max(min(mem1[i]-1,1),0)
                        mem1[i+1:i+2]+=delta
                    else:
                        cost[i+1] = cost[i] + max(min(1-mem1[i],1),0)
                        mem1[i+1:i+2]-=mem1[i]*(1-1/tau_m)
                    paired_flag = False
    return cost[-1]

def index_to_spike_train(index, time_steps):
    return np.array(['0']*(time_steps-len(list(bin(index)[2:])))+list(bin(index)[2:])).astype('int')
def spike_train_to_index(spike_train):
    return int('0b'+''.join(str(np.array(spike_train).astype('int'))[1:-1].split(' ')),0)
spike_dist_matrix = np.load('figs/spike_dist_matrix.npy')
spike_transfer_matrix_v4 = np.zeros_like(spike_dist_matrix)
for i in range(2,99):
    print("iters:",i)
    out = np.load('figs/out'+str(i)+'.npy',allow_pickle=True)
    out = out.item()
    mem = np.load('figs/mem'+str(i)+'.npy',allow_pickle=True)
    mem = mem.item()
    out_next = np.load('figs/out'+str(i+1)+'.npy',allow_pickle=True)
    out_next = out_next.item()
    for key in out:
        if key == 'conv_2':
            layer_shape = np.shape(np.array(out[key]))
            cases_num = layer_shape[0]*layer_shape[1]*layer_shape[2]*layer_shape[3]
            output_next = np.array(out_next[key]).reshape(cases_num,layer_shape[4])
            output_ori = np.array(out[key]).reshape(cases_num,layer_shape[4])
            mem_ori = np.array(mem[key]).reshape(cases_num,layer_shape[4])
            for index_case in range(len(output_ori)):
                case = output_ori[index_case]
                new_case = output_next[index_case]
                if np.sum(case-new_case) == 0:
                    continue
                mem_case = mem_ori[index_case]
                ori_pattern = spike_train_to_index(case)
                new_pattern = spike_train_to_index(new_case)
                ranking = [0]*1024
                for cal_dist in range(1024):
                    cal_spike_train = index_to_spike_train(cal_dist, 10)
                    ranking[cal_dist] = spike_distance_v4(case.copy(),cal_spike_train,mem_case.copy())
                v4_rank = np.argsort(ranking).tolist().index(new_pattern)
                spike_transfer_matrix_v4[ori_pattern, v4_rank] += 1
    np.save('figs/spike_transfer_matrix_v4'+str(i), spike_transfer_matrix_v4)
