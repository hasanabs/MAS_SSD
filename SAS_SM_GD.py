"""
@author: Hasan Albinsaid
@site: https://github.com/hasanabs
"""
import time
import numpy as np
import fungsi_ris as fn

N,Nr,M=64,16,4
Ns=10000
SNR_Min,step,SNR_Max=-32,4,0;
Es=1

L1=np.int(np.log2(Nr))
L2=np.int(np.log2(M))
L=L1+L2
Transmit_bit=L*Ns
modulation= fn.modulation(M)

print("N."+str(N)+" Nr."+str(Nr)+" Ns."+str(Ns)+" M."+str(M))
start = time.time()
Range=np.arange(SNR_Min,SNR_Max+1,step);
Error_bit=np.zeros((len(Range), 1), dtype=np.float16)
index_error=0
predict_stop=0
for SNR_dB in Range:
    Err_acumulation=0;
    #Sending data   
    for send in range(Ns):
        data=np.random.randint(2, size=L)
        x=modulation[fn.bi2de(data[L1:L])]
        H=fn.H(Nr,N)
        phi=np.angle(H[fn.bi2de(data[0:L1]),:])
        exp_phi=np.reshape(np.exp(-1j*phi),[N,1])
        noise=fn.noise(SNR_dB, Nr, Es)
        r=np.matmul(H,exp_phi)*x+noise
        
        #decode
        predict_start = time.time()
        m_hat=np.argmax(np.abs(r))
        phi_GD=np.angle(H[m_hat,:])
        exp_phi=np.reshape(np.exp(-1j*phi_GD),[N,1])  
        x_hat=np.argmin(abs(r[m_hat]-np.matmul(H[m_hat,:],exp_phi)*modulation))
        decoded=np.append(fn.de2bi((m_hat),L1)[0],fn.de2bi((x_hat),L2)[0])
        Err_acumulation+=(data!=decoded).sum()
        predict_stop = predict_stop + time.time()-predict_start

    Error_bit[index_error,0]=Err_acumulation/(send*L)
    print("SNR="+str(SNR_dB)+" n_Error="+str(Err_acumulation)+" transmited="+str(send*L)+" BER="+str(Error_bit[index_error,0]))
    index_error=index_error+1
    
#Plot
Title = "N="+str(N)+"  Nr="+str(Nr)+" M="+str(M)+"  Ns="+str(Ns)
Label = "SAS-SM"
fn.plotter(Range, Error_bit, SNR_Min, SNR_Max, L, 'rv-', Title, Label)
print ("Time : ", time.time()-start, "seconds.")
print ("Time Complexity: ", predict_stop, "seconds.")