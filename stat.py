import numpy as np
import torch

# This function finds the nearest neighbor under the metric: distance_v3
# The two inputs are the current time step's spike train output, and the current membrane potential
# The return valur is the nearest neighbor
# Expecting data format is torch.tensor
def find_smallest_one(current_output, membrane_potential):
    time_steps = len(current_output[0,0,0,0])
    max_dist = 0.2
    flip_distance = torch.clamp(torch.abs(1-membrane_potential),0,1)
    sorted_index = torch.argsort(flip_distance)
    dim1 = [val for val in list(range(len(flip_distance))) for i in range(len(flip_distance[0]))] 
    dim2 = list(range(len(flip_distance[0])))*len(flip_distance)
    dim3 = dim4 = [0]*len(dim1)
    dim5 = sorted_index[:,:,:,:,0].reshape(len(dim1))
    new_output = current_output.clone().detach()
    near_by = flip_distance[dim1,dim2,dim3,dim4,dim5]<max_dist
    while near_by.any() == True:
        new_output[dim1,dim2,dim3,dim4,dim5]=new_output[dim1,dim2,dim3,dim4,dim5]^near_by
        near_by = near_by & (dim5!=time_steps-1)
        dim5 = torch.clamp(dim5+1,0,time_steps-1)
        nopp = new_output[dim1,dim2,dim3,dim4,dim5-1] # new output previous pointer
        mpp = membrane_potential[dim1,dim2,dim3,dim4,dim5] # membrane potential pointer
        changes = (near_by&(nopp==1)&((1<=mpp)&(mpp<1.8)))|\
                    (near_by&(nopp==0)&((0.2<mpp)&(mpp<1)))
        near_by = near_by & changes
    return new_output

def spike_distance_v1(str1, str2):
    # str1 is the original spike sequence
    # str2 is the new spike sequence
    assert(len(str1)==len(str2)!=0)
    moves=vanish_spikes=new_spikes=[]
    cost = [0] * (len(str1) + 1)
    paired_flag = False # prevent pairs overlap
    for i in range(len(str1)):
        if i == 0:
            if str1[i] != str2[i]:
                cost[1] = 1
            continue
        if str1[i] != str2[i]:
            if paired_flag == False and str1[i-1] == str2[i] and str1[i] == str2[i-1]:
                cost[i+1] = cost[i-1] + 1
                if str1[i] == 1:
                    moves = moves + [(i,i-1)]
                    new_spikes = new_spikes[:-1]
                else:
                    moves = moves + [(i-1,i)]
                    vanish_spikes = vanish_spikes[:-1]
                paired_flag = True
            else:
                cost[i+1] = cost[i] + 1
                if str1[i] == 1:
                    vanish_spikes = vanish_spikes + [i]
                else:
                    new_spikes = new_spikes + [i]
                paired_flag = False
        else:
            cost[i+1] = cost[i]
    return cost[-1], moves, new_spikes, vanish_spikes

def spike_distance_matrix_v1(ori_str_matrix, next_str_matrix):
    assert(np.array(ori_str_matrix).shape == np.array(next_str_matrix).shape)
    color_matrix = ori_str_matrix
    for i in range(len(ori_str_matrix)):
        color_matrix[i] = ori_str_matrix[i]
        dis, pairs, new_spikes, vanish_spikes = spike_distance_v1(ori_str_matrix[i], next_str_matrix[i])
#         if len(vanish_spikes) != 0:
#             print(dis,pairs,new_spikes,vanish_spikes)
        for pair in pairs:
            color_matrix[i][pair[0]]=0.8
            color_matrix[i][pair[1]]=0.9
        for new_spike in new_spikes:
            color_matrix[i][new_spike]=0.5
        for vanish_spike in vanish_spikes:
            color_matrix[i][vanish_spike] = 0.3
    return color_matrix

def index_to_spike_train(index, time_steps):
    return np.array(['0']*(time_steps-len(list(bin(index)[2:])))+list(bin(index)[2:])).astype('int')

def spike_train_to_index(spike_train):
    return int('0b'+''.join(str(np.array(spike_train).astype('int'))[1:-1].split(' ')),0)

def spike_distance_v2(str1, str2, mem1):
    # str1 is the original spike sequence, mem1 is the original membrane potential
    # str2 is the new spike sequence
    assert(len(str1)==len(str2)!=0)
    cost = [0] * (len(str1) + 1)
    paired_flag = False
    for i in range(len(str1)):
        if i == 0:
            if str1[0] != str2[0]:
                if str1[0] == 1:
                    cost[1] = min(1, mem1[0]-1)
                else:
                    cost[1] = min(1, 1-mem1[0]) # Do not need the 'min' for '1-mem1[i]', if can garantee mem>0
            continue
        if str1[i] != str2[i]:
            if paired_flag == False and str1[i-1] == str2[i] and str1[i] == str2[i-1]:
                if str1[i] == 1:
                    cost[i+1] = cost[i-1] + (min(1, mem1[i]-1) + min(1, 1-mem1[i-1]))/2
                else:
                    cost[i+1] = cost[i-1] + (min(1, 1-mem1[i]) + min(1, mem1[i-1]-1))/2
                paired_flag = True
            else:
                if str1[i] == 1:
                    cost[i+1] = cost[i] + min(1, mem1[i]-1)
                else:
                    cost[i+1] = cost[i] + min(1, 1-mem1[i])
                paired_flag = False
        else:
            cost[i+1] = cost[i]
    return cost[-1]

def dist_v1_plot():
    spike_change_matrix = np.load('figs/spike_change_matrix.npy')
    spike_dist_matrix = np.load('figs/spike_dist_matrix.npy')

    sums = np.zeros(11)

    for i in range(1024):
        spike_change_matrix[i,i]=0
    for i, line in enumerate(spike_dist_matrix):
        order = np.argsort(line,kind='stable')
        spike_change_matrix[i] = spike_change_matrix[i][order]
        spike_dist_matrix[i] = spike_dist_matrix[i][order]

    for line_index, line in enumerate(spike_change_matrix):
        for colum_index, spike in enumerate(line):
            dist = spike_dist_matrix[line_index][colum_index]

            sums[int(dist)]+=spike
    print(sums)

    total=np.zeros(1024)
    for line in spike_change_matrix:
        total = total + np.array(line)

    sums = 0
    accum = np.zeros(1024)
    for index,i in enumerate(total):
        sums += i
        accum[index] = sums
    accum = accum/sums*100

    plt.figure(figsize=(24,10))
    plt.subplot(2,3,1)
    plt.imshow(spike_dist_matrix,'Blues',alpha=1)
    plt.colorbar()
    plt.text(200,412,'Distance background',fontsize=15)

    locs=[10,50,150,400,650,850,970,1000,1023]
    for i in range(len(locs)):
        plt.text(locs[i],512,str(int(spike_dist_matrix[512,locs[i]])))

    plt.xlabel('Spike-trains\' Distant ranking',fontsize=15)
    plt.ylabel('1024 Spike-trains\' binary index',fontsize=15)


    plt.subplot(2,3,2)
    plt.plot(total)
    plt.xlabel('Spike-trains\' Distant ranking',fontsize=15)
    plt.ylabel('Number of transfermation',fontsize=15)
    plt.title('Number of transfermation V.S. Distance metric version1', fontsize=15)

    plt.subplot(2,3,3)
    plt.plot(accum, marker='+',color='r', markersize=2, mec='blue')
    plt.xlabel('Spike-trains\' Distant ranking',fontsize=15)
    plt.ylabel('Accumulated probability',fontsize=15)

    plt.subplot(2,3,4)
    plt.imshow(np.log(np.array(spike_change_matrix)+1), cmap='Reds',alpha=0.8)
    plt.colorbar()
    plt.xlabel('Spike-trains\' Distant ranking',fontsize=15)
    plt.ylabel('1024 Spike-trains\' binary index',fontsize=15)
    plt.text(250,512,'Natural logarithm',fontsize=15)
    plt.text(170,612,'of the transfer counts',fontsize=15)

    plt.subplot(2,3,5)
    plt.plot(total[:50])
    plt.xlabel('Spike-trains\' Distant ranking',fontsize=15)
    plt.ylabel('Number of transfermation',fontsize=15)
    plt.text(10,80000,'X-axis zoom in of the above plot',fontsize=15)
    plt.subplot(2,3,6)
    plt.plot(accum[:120],marker='+',color='r', markersize=4, mec='blue')
    plt.xlabel('Spike-trains\' Distant ranking',fontsize=15)
    plt.ylabel('Accumulated probability',fontsize=15)
    plt.text(20,50,'X-axis zoom in of the above plot',fontsize=15)

def dist_v2_plot():
    spike_transfer_matrix_v2 = np.load('figs/spike_transfer_matrix_v2.npy')
    total=np.zeros_like(spike_transfer_matrix_v2[0])
    for i in spike_transfer_matrix_v2:
        total = total + np.array(i)
    plt.figure(figsize = (24,10))

    plt.subplot(2,3,1)
    plt.imshow(spike_transfer_matrix_v2, cmap='Reds')
    plt.colorbar()
    plt.xlabel('Spike-trains\' Distant ranking',fontsize=15)
    plt.ylabel('1024 Spike-trains\' binary index',fontsize=15)

    plt.subplot(2,3,2)
    plt.plot(total)
    plt.xlabel('Spike-trains\' Distant ranking',fontsize=15)
    plt.ylabel('Number of transfermation',fontsize=15)
    plt.title('Number of transfermation V.S. Distance metric version2', fontsize=15)

    accumulate = [0]*1024
    sums = 0
    for index, num in enumerate(total):
        sums = sums + num
        accumulate[index] = sums
    accumulate = np.array(accumulate)/sums * 100
    plt.subplot(2,3,3)
    plt.plot(accumulate,marker='+',color='r', markersize=2, mec='blue')
    plt.xlabel('Spike-trains\' Distant ranking',fontsize=15)
    plt.ylabel('Accumulated probability',fontsize=15)

    plt.subplot(2,3,4)
    plt.imshow(np.log(np.array(spike_transfer_matrix_v2)+1), cmap='Reds')
    plt.colorbar()
    plt.xlabel('Spike-trains\' Distant ranking',fontsize=15)
    plt.ylabel('1024 Spike-trains\' binary index',fontsize=15)
    plt.text(200,512,'Natural logarithm',fontsize=15,color='r')
    plt.text(200,612,'of the above plot',fontsize=15,color='r')

    plt.subplot(2,3,5)
    plt.plot(total[:50])
    plt.xlabel('Spike-trains\' Distant ranking',fontsize=15)
    plt.ylabel('Number of transfermation',fontsize=15)
    plt.text(10,260000,'X-axis zoom in of the above plot',fontsize=15)
    plt.subplot(2,3,6)
    plt.plot(accumulate[:120],marker='+',color='r', markersize=4, mec='blue')
    plt.xlabel('Spike-trains\' Distant ranking',fontsize=15)
    plt.ylabel('Accumulated probability',fontsize=15)
    plt.text(20,50,'X-axis zoom in of the above plot',fontsize=15)
