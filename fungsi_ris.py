"""
@author: Hasan Albinsaid
@site: https://github.com/hasanabs
"""
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

def nck(n,k):
    return np.math.factorial(n)/np.math.factorial(k)/np.math.factorial(n-k)

def nchoosek(arr, k):
    return np.array(list(itertools.combinations(arr, k)))

def optimum_RAC(all_RAC, n, r, size_comb):
    ukuran=np.zeros(n,dtype=int)
    while(len(all_RAC)>size_comb):
        for i in range(n):
            ukuran[i]=(all_RAC==i+1).sum()
        idx_rem=0;
        remaining_idx=np.arange(len(all_RAC))
        sort_remove=np.argsort(-ukuran)
        while(len(remaining_idx)>1):
            old_remaining_idx=remaining_idx
            remaining_idx=remaining_idx[np.where((all_RAC[remaining_idx,:]==sort_remove[idx_rem]+1))[0]]
            if (len(remaining_idx)==0):
                idx=0
                while(len(remaining_idx)==0):
                   remaining_idx=old_remaining_idx[np.where((all_RAC[old_remaining_idx,:]==sort_remove[idx]+1))[0]]
                   idx+=1
            idx_rem+=1
        all_RAC=np.delete(all_RAC, (remaining_idx), axis=0)
    return all_RAC

def bi2de(arr):
    result=0
    for i in range(len(arr)):result+=np.power(2,i)*arr[len(arr)-1-i]
    return result

def de2bi(decimal, L_bit):
    arr=np.zeros((1,L_bit), dtype=np.int8)
    for i in range(L_bit): 
        arr[0,(L_bit-i-1)]=decimal%2
        decimal=decimal>>1
    return arr

def modulation(M):
    if M==2: modulation=np.array([-1+0j, 1+0j])
    elif M==4: modulation=np.array([-1-1j, -1+1j, 1+1j, 1-1j]/np.sqrt(2))
    elif M==16: modulation=np.array([-3+3j, -3+1j, -3-3j, -3-1j, 
                                 -1+3j, -1+1j, -1-3j, -1-1j,
                                  3+3j,  3+1j,  3-3j,  3-1j,
                                  1+3j,  1+1j,  1-3j,  1-1j]/np.sqrt(10))
    return modulation

def herm(matrix):
    return np.transpose(np.conjugate(matrix))
 
def H(Nr, Nt):
    return (np.random.randn(Nr,Nt)+np.random.randn(Nr,Nt)*1j)/np.sqrt(2)
    
def noise(SNR, Nr, Es):
    return (np.random.randn(Nr,1)+np.random.randn(Nr,1)*1j)*np.sqrt(Es/np.power(10,(SNR)/10))/np.sqrt(2)

def plotter(Range, Error_bit, SNR_Min, SNR_Max, L, prop, Title, Label):
    plt.figure(1)
    ASBT = (np.ones((len(Error_bit),1)) - Error_bit)*L
    plt.plot(Range, ASBT, prop, linewidth=1, label=Label)
    plt.legend(loc='lower right', fontsize='x-large')
    plt.axis([SNR_Min, SNR_Max, 2, 10.5])
    plt.yscale('linear')
    plt.xlabel('SNR[dB]')
    plt.ylabel('ASBT')
    plt.minorticks_on()
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor',alpha=0.4)
    plt.suptitle('ASBT '+ Label, fontsize='x-large', fontweight='bold')
    plt.title(Title, fontsize='large', fontweight='book')
    plt.show()    
    if not os.path.exists('../results'): os.makedirs('../results')
    plt.savefig('../results/ASBT_'+Label+'.png')
    
    plt.figure(2)
    plt.plot(Range, Error_bit, prop, linewidth=1, label=Label)
    plt.legend(loc='upper right', fontsize='x-large')
    plt.axis([SNR_Min, SNR_Max, 6e-4, 1e-0])
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('SNR[dB]')
    plt.ylabel('BER')
    plt.minorticks_on()
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor',alpha=0.4)
    plt.suptitle('BER ' + Label, fontsize='x-large', fontweight='bold')
    plt.title(Title, fontsize='large', fontweight='book')
    plt.show()    
    if not os.path.exists('../results'): os.makedirs('../results')
    plt.savefig('../results/'+Label+'.png')
    
    