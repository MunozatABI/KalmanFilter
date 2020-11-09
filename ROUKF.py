# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:36:05 2020

@author: lmun373

Sources: http://hemolab.lncc.br/producao/2017_thesis_gonzalo_daniel_maso_talou.pdf
         https://gitlab.com/gmasotalou/kalman/-/tree/master

"""

import numpy as np
import brian2 as b2

def starSigmaPoints(nParameters):
    nSigmas = 2 * nParameters + 1
    sigmas = np.zeros([nParameters, nSigmas])
    np.fill_diagonal(sigmas, (np.sqrt((2. * nParameters + 1.)/2.)))
    sigmas[:, nParameters:nSigmas-1] = np.flip(sigmas[:, 0: nParameters] * -1, 1)

    return sigmas 

class reduced_UKF:
    def __init__(self, nObservations, nStates, nParameters, observationsUncertainty, parametersUncertainty):
        
        self.nObservations = nObservations 
        self.nStates = nStates
        self.nParameters = nParameters 
        self.observationsUncertainty = observationsUncertainty 
        self.parametersUncertainty = parametersUncertainty
        
        self.X = np.zeros([nStates, 1])
        self.Theta = np.zeros([nParameters, 1])
        
        self.LTheta = np.eye(nParameters, nParameters)
        self.LX = np.zeros([nStates, nParameters])
        self.U = np.eye(nParameters, nParameters)
        self.Wi = np.eye(nObservations, nObservations)
        
        np.fill_diagonal(self.U, 1./parametersUncertainty)
        np.fill_diagonal(self.Wi, 1./observationsUncertainty)
        
        self.sigma = starSigmaPoints(nParameters)
        self.alpha = 1. / (np.shape(self.sigma)[1])
        
        Dsigma = self.alpha * np.eye(np.shape(self.sigma)[1], np.shape(self.sigma)[1]) #Size of Dsignma array (nsigmapoints, nsigmapoints)
        self.Dsigma = np.matmul(Dsigma,self.sigma.T)
        self.Pa = np.matmul(self.sigma,self.Dsigma)

    def executeStep(self, zkhat, my_A, my_H):
        ''' A is the forward operation
            H is the observation operation'''
            
        #Matrices
        Thetak = np.zeros([self.nParameters, np.shape(self.sigma)[1]])
        Xk = np.zeros([self.nStates, np.shape(self.sigma)[1]])
        Zk = np.zeros([self.nObservations, np.shape(self.sigma)[1]])
        HL = []
        invU = np.linalg.inv(self.U)
        C = np.linalg.cholesky(invU)
    
    	#Initialise Column vectors
        thetak = 1 
        xk = 1
    
        for i in range(np.shape(self.sigma)[1]):
    		#Sample Sigma point
            s = self.sigma[:,i].reshape(self.nParameters,1)
                 
            #Prediction Step (sigma points maintain estimate for states, sample across dimensions of parameters theta - Eqns 6.2.45)
            xk = self.X + np.matmul(self.LX, np.matmul(C.T,s))
            thetak = self.Theta + np.matmul(self.LTheta,np.matmul(C.T,s))

    		#Propagate sigma point
            xk, thetak = my_A(xk, thetak)
            
    		#Perform observation
            zk = my_H(xk)
            
    		#Transform theta_k to assimilation values
            Xk[:,i] = xk
            Thetak[:,i] = thetak.reshape(self.nParameters,)
            Zk[:,i] = zk
	
        #New state and its associated observation
        xk = np.mean(Xk, axis=1)
        thetak = np.mean(Thetak, axis=1)
        zkMean = np.mean(Zk, axis=1)	#Only constant alpha
        error = zkhat - zkMean
        error = error.reshape(2,1)
        
        #Correction step - Eqn 6.2.46
    	#Update covariance matrixes
        self.LX = np.matmul(Xk,self.Dsigma) #Xk shape = 10000,3   Dsigma shape = 3,3
        self.LTheta = np.matmul(Thetak,self.Dsigma)
        HL = np.matmul(Zk,self.Dsigma)
        self.U = self.Pa + np.matmul(HL.T,np.matmul(self.Wi,HL))
        
        invU = np.linalg.inv(self.U) #Singular Matrix error
    
    	#Compute new estimate
        self.X = xk + np.matmul(self.LX,np.matmul(invU,np.matmul(HL.T,np.matmul(self.Wi,error))))
        self.Theta = thetak + np.matmul(self.LTheta,np.matmul(invU,np.matmul(HL.T,np.matmul(self.Wi,error))))
        
        currError = np.linalg.norm(error, 'fro')
        #prevError = currError #Why do I need prevError?
    	#++currIt ### Where does currIt come from?
        
        return currError

def my_A(state_vector, parameter_vector):
    # state transition function - predict next state based
    # on spiking neuron model
    weight = parameter_vector
    #print(weight)
    b2.start_scope()
    # Parameters
    num_inputs = 1
    input_rate = 100*b2.Hz
    b2.prefs.codegen.target = 'numpy'
   #input_rate = [1, 2, 3, 4, 5]*b2.Hz
    tau = 1*b2.ms
    P = b2.PoissonGroup(num_inputs, rates=input_rate)
    eqs = '''
    dv/dt = -v/tau : 1
    w : 1
    '''
    G = b2.NeuronGroup(1, eqs, threshold='v>1', reset='v=0', method='exact')
    S = b2.Synapses(P, G, on_pre='v += w')
    S.connect(p = 1)
    S.w = weight[0]
    Mon = b2.StateMonitor(G, 'v', record=True)
    net = b2.Network(b2.collect())        
    net.run(1*b2.second)
    new_state_vector = Mon.v[0][0:3]
    #len(Mon.v[0])) = 10000
    return new_state_vector, weight

def my_H(state_vector):
    # measurement function - convert state [voltages, weights] into a measurement
    # where measurement is the maximum voltage
    voltages = state_vector
    v_max = np.max(voltages)
    return v_max

def main():
    test = reduced_UKF(2, 3, 2, 1, 1)
    error = test.executeStep([5, 5], my_A, my_H)
    print('Error:', error)

if __name__=='__main__':
    main()

