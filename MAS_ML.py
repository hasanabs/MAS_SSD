"""
@author: Hasan Albinsaid
@site: https://github.com/hasanabs
"""
import time
import numpy as np
import fungsi_ris as fn

N,Nr,Np,M=64,12,2,4;
Ns=10000
SNR_Min,step,SNR_Max=-32,4,0;
Es=Np
sqrt_Es=np.sqrt(Es)
alpha=[0.2,0.8] #Alpha is power allocation factor, and it must writen in ascending order

L1= np.int(np.floor(np.log2(fn.nck(Nr,Np))))
L2=Np*np.int(np.log2(M))
L=L1+L2
Transmit_bit=L*Ns
signal_power=np.sqrt(Es)
modulation= fn.modulation(M)
bit_per_sym=np.int(np.sqrt(M))

#RAC Initialization
C= np.int(np.power(2,L1))
prob_RAC=fn.nchoosek(np.arange(1,Nr+1),Np)
R=fn.optimum_RAC(prob_RAC,Nr,Np,C)

print("N."+str(N)+" Nr."+str(Nr)+" Np."+str(Np)+" M."+str(M)+" Ns."+str(Ns))
start = time.time()
Range=np.arange(SNR_Min,SNR_Max+1,step);
Error_bit=np.zeros((len(Range), 1), dtype=np.float16)
index_error=0
predict_stop=0
for SNR_dB in Range:
    Err_acumulation=0;
    sigma_sq=Es/pow(10,(SNR_dB)/10);
    #Sending data   
    for send in range(Ns):
        r=np.zeros([Nr,1],dtype=complex)
        H_dev=np.zeros([N,1],dtype=complex)
        data=np.random.randint(2, size=L)
        H=fn.H(Nr,N)
        active_H=H[R[fn.bi2de(data[0:L1]),:]-1,:]
        p_alloc_sort=np.argsort(-abs(np.sum(active_H,axis=1))/sum(abs(np.sum(active_H,axis=1)))) #Power allocation index for each channel
        x=0;
        for i in range(Np):
            x=x+Es*np.sqrt(alpha[p_alloc_sort[i]])*modulation[fn.bi2de(data[L1+bit_per_sym*i:L1+bit_per_sym*(i+1)])]
        for i in range(int(N)):
            H_dev[i,0]=active_H[i%Np,i]
        phi=np.angle(H_dev)
        exp_phi=np.exp(-1j*phi)
        noise=fn.noise(SNR_dB, Nr, Es)
        r=np.matmul(H,exp_phi)*x+noise
        
        
        #decode
        predict_start = time.time()
        min_distance=5000
        final_candidate=0
        for cek in range(np.power(2,L1)):
            rac_sellected=R[cek,:]-1
            active_H=H[rac_sellected,:]
            for i in range(int(N)):
                H_dev[i,0]=active_H[i%Np,i]
            phi_ML=np.angle(H_dev)
            exp_phi=np.reshape(np.exp(-1j*phi_ML),[N,1])
            #ML
            H_theta=np.matmul(H,exp_phi);
            for mod in range (np.power(2,L2)):
                dat_cek = fn.de2bi(mod,L2)[0]
                s=0;
                s_hat=np.zeros([Np,1],dtype=complex)
                for i in range(Np):
                    s_hat[i]=modulation[fn.bi2de(dat_cek[bit_per_sym*i:bit_per_sym*(i+1)])]
                    s=s+Es*np.sqrt(alpha[p_alloc_sort[i]])*s_hat[i]
                compare=np.linalg.norm(r-H_theta*s, 'fro')
                if min_distance>compare:
                    min_distance=compare
                    final_candidate=cek
                    x_hat=s_hat
        decoded=np.zeros(L,dtype=int)
        decoded[0:L1]=fn.de2bi(final_candidate,L1)[0]
        for i in range(Np):
            decoded[L1+i*bit_per_sym:L1+(i+1)*bit_per_sym]=fn.de2bi(np.argmin(abs(modulation-x_hat[i])),bit_per_sym)[0]
        Err_acumulation+=(data!=decoded).sum()
        predict_stop = predict_stop + time.time()-predict_start

    Error_bit[index_error,0]=Err_acumulation/(Ns*L)
    print("SNR="+str(SNR_dB)+" n_Error="+str(Err_acumulation)+" transmited="+str(Ns*L)+" BER="+str(Error_bit[index_error,0]))
    index_error=index_error+1
    
#Plot
Title = "N="+str(N)+"  Nr="+str(Nr)+"  Np="+str(Np)+" M="+str(M)+"  Ns="+str(Ns)
Label = "MAS-ML"
fn.plotter(Range, Error_bit, SNR_Min, SNR_Max, L, 'kv-', Title, Label)
print ("Time : ", time.time()-start, "seconds.")
print ("Time Complexity: ", predict_stop, "seconds.")