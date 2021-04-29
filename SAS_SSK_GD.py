"""
@author: Hasan Albinsaid
@site: https://github.com/hasanabs
"""
import time
import numpy as np
import fungsi_ris as fn

N,Nr=64,16
Ns=10000
SNR_Min,step,SNR_Max=-32,4,0;
Es=1
sqrt_Es=np.sqrt(Es)

L1= np.int(np.log2(Nr))
Transmit_bit=L1*Ns
signal_power=np.sqrt(Es)

print("N."+str(N)+" Nr."+str(Nr)+" Ns."+str(Ns))
start = time.time()
Range=np.arange(SNR_Min,SNR_Max+1,step);
Error_bit=np.zeros((len(Range), 1), dtype=np.float16)
index_error=0
predict_stop=0
for SNR_dB in Range:
    Err_acumulation=0;
    #Sending data   
    for send in range(Ns):
        data=np.random.randint(2, size=L1)
        H=fn.H(Nr,N)
        phi=np.angle(H[fn.bi2de(data),:])
        exp_phi=np.reshape(np.exp(-1j*phi),[N,1])
        noise=fn.noise(SNR_dB, Nr, Es)
        r=sqrt_Es*np.matmul(H,exp_phi)+noise
        #decode
        predict_start = time.time()
        decoded=fn.de2bi(np.argmax(abs(r)),L1)[0]
        Err_acumulation+=(data!=decoded).sum()
        predict_stop = predict_stop + time.time()-predict_start

    Error_bit[index_error,0]=Err_acumulation/(send*L1)
    print("SNR="+str(SNR_dB)+" n_Error="+str(Err_acumulation)+" transmited="+str(send*L1)+" BER="+str(Error_bit[index_error,0]))
    index_error=index_error+1
    
#Plot
Title = "N="+str(N)+"  Nr="+str(Nr)+"  Ns="+str(Ns)
Label = "SAS-SSK"
fn.plotter(Range, Error_bit, SNR_Min, SNR_Max, L1, 'bv-', Title, Label)
print ("Time : ", time.time()-start, "seconds.")
print ("Time Complexity: ", predict_stop, "seconds.")