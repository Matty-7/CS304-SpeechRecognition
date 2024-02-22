#class of mix info from KMeans
#Import packages
import IPython.display as ipd
import numpy as np
import pyaudio, wave
import math
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from scipy.fftpack import dct

def getMFCC2(wavename):#without normalization
    import numpy as np
    import scipy.io.wavfile as wav
    from python_speech_features import mfcc
    fs, audio = wav.read(wavename)
    feature_mfcc = mfcc(audio, samplerate=fs)
    mfcc=[]
    mfcc.append(np.hstack([feature_mfcc[0],feature_mfcc[0],feature_mfcc[0]]))
    for i in range(1,len(feature_mfcc)-1):
        delta=np.zeros(13)
        for j in range(13):
            delta[j]=feature_mfcc[i+1][j]-feature_mfcc[i-1][j]
        mfcc.append(np.hstack([feature_mfcc[i],delta]))
    mfcc.append(np.hstack([feature_mfcc[-1],feature_mfcc[-1],feature_mfcc[-1]]))

    for i in range(1,len(mfcc)-1):
        acc=np.zeros(13)
        for j in range(13):
            acc[j]=mfcc[i+1][13+j]-mfcc[i-1][13+j]
        mfcc[i]=np.hstack([mfcc[i],acc])
    mfccs=np.array(mfcc)
    std=np.std(mfccs)
    var=np.var(mfccs,1)
    for i in range(len(mfccs)):
        for j in range(39):
            mfccs[i][j]=mfccs[i][j]/var[i]
    return mfccs


class mixInfo:
    """docstring for mixInfo"""
    def __init__(self):
        self.Gaussian_mean = []#每个gaussian distribution的 mean vector
        self.Gaussian_var = [] #每个gaussian distribution的 diagnol covarience
        self.Gaussian_weight = []#每个gaussian distribution 的权重（其和为1）
        self.Num_of_Gaussian = 0 #几个gaussian distribution
class hmmInfo:
    '''hmm model param'''
    def __init__(self):
        self.init = [] #初始矩阵
        self.transition_cost = []
        self.mix = [] #高斯混合模型参数,有几个state，里面就有几个mix
        self.N = 0 #状态数

def log_gaussian(mu,squared_sigma,input_vector):
    
    #Calculate the cost using log gaussian
    part1=0.5*np.sum(np.log((2*np.pi)*(squared_sigma)),axis=1)
    part2=0.5*np.sum(np.square((input_vector-mu))/squared_sigma,axis=1)
    cost= part1+part2
    return cost

def gaussian(mu,squared_sigma,input_vector):
    
    #Calculate the probability, we only return a number!!!!
    #为了方便 我这边的一个numpy 推广就先应用到mu 上吧 毕竟我后面是一帧一帧的去分析
    #print(type(squared_sigma))
    #d=input_vector.shape[0]
    d=2
    part0=np.prod(squared_sigma,axis=1)
    part1=np.sqrt((2*np.pi)**d *part0)
    front=1/part1
    part2=0.5*np.sum((mu-input_vector)**2/squared_sigma,axis=1)
    expo=np.exp(-part2)
    p=front*expo
    #p=np.exp(log_gaussian(mu,squared_sigma,x))
    return p

def mixture_log_gaussian(mix,input_vector):
    weight=mix.Gaussian_weight
    mu = mix.Gaussian_mean
    squared_sigma = mix.Gaussian_var
    cost=log_gaussian(mu,squared_sigma,input_vector)
#     print(cost)
#     print(weight)
    weighted_cost=np.sum(weight*cost)
    return weighted_cost


def traceback(D):
    #start from the last state and last frame
    current_state,current_frame=np.array(D.shape)-1
    #insert the last frame's state
    x=[current_state]
    
    #print(current_frame+1)
    # we do not need the frame 0, which is the fine
    while current_state>0 and current_frame>1:
        #move to the previous frame
        current_frame-=1
        #print(current_state)
        if current_state>2:
            to_check=[D[current_state][current_frame-1],
                      D[current_state-1][current_frame-1],
                      D[current_state-2][current_frame-1]]
            track=np.argmin(to_check)
            if track==2:
                print("跳跃两state？？？？")
                print(to_check)
                print(current_frame)
        elif current_state>1:
            to_check=[D[current_state][current_frame-1],
                      D[current_state-1][current_frame-1]]
            track=np.argmin(to_check)
        else:
            track=0
            
        if track==0:
            #which means, last frame still in the same stage
            x.insert(0,current_state)
        elif track==1:
            current_state-=1
            x.insert(0,current_state)
        else:
            current_state-=2
            x.insert(0,current_state)
    #print(x)
    return x

class GMMHMM(object):
    #please input all the templates for one specific word
    def __init__(self,templates,Gaussian_distribution_number=[4,4,4,4,4]):
        self.templates=templates
        #len should be state_number
        self.Gaussian_distribution_number=Gaussian_distribution_number
        self.state_number=len(self.Gaussian_distribution_number)
        self.node_in_each_state=[]
        self.node_state=[]
        self.hmm =None
    
    def update_node_in_each_state(self,show_result=False):
        self.node_in_each_state=[]#state number decide sublist number
        # self.node_in_each_state[0] is empty
        for state in range(self.state_number+1):
            self.node_in_each_state.append([])

        for k in range(len(self.templates)):#templates number
            # the i th vector of the k th training sequence
            for i in range(len(self.node_state[k])):
                j=int(self.node_state[k][i])#the state of the i th vector
                self.node_in_each_state[j].append(self.templates[k][i])
#         if show_result:
#             num_nodes=[]
#             for state_nodes in self.node_in_each_state:
#                 num_nodes.append(len(state_nodes))
#             print()
#             print("Current num of nodes in each state is shown as below {}".format(num_nodes))
    
    def compute_transition_cost(self,show_result=False):
        # 6 PPT p.g 53  return the transition cost matrix
        shift_likehood=np.zeros((self.state_number+1,self.state_number+1))
        self.state_node_num=np.zeros(self.state_number+1)
        #fetch all the initial state
        initial_states=[]
        for k in range(len(self.node_state)):
            shift_likehood[0][self.node_state[k][0]]+=1

       # count the state transition of all the nodes
        for k in range(len(self.node_state)):
            for i in range(len(self.node_state[k])-1):
                current_node=self.node_state[k][i]
                next_node=self.node_state[k][i+1]
                shift_likehood[current_node][next_node]+=1 
                self.state_node_num[current_node]+=1
            #last node case        
            shift_likehood[self.node_state[k][-2]][self.node_state[k][-1]]+=1
            self.state_node_num[self.node_state[k][-1]]+=1
        
        if show_result:
            print("The num of nodes in different states are {}".format(self.state_node_num))
        #It is sometimes useful to permit entry directly into later states
        for j in range(self.state_number+1):
            #N is the total number of training sequences
            N=len(self.node_state)
            #N_0j is the number of training sequences for which
            #the first data vector was in the j th state
            N_0j=shift_likehood[0][j]
            shift_likehood[0][j]=N_0j/N
            if N_0j==0:
                shift_likehood[0][j]=np.inf
            else:
                shift_likehood[0][j]=-np.log(shift_likehood[0][j])

        #6 PPT p.g 55
        for j in range(1,self.state_number+1):
            for k in range(j,self.state_number+1):
                shift_likehood[j][k]=shift_likehood[j][k]/self.state_node_num[j]
                #transition probability---->transition cost
                #T_ij in 6 PPT p.g 58
                if shift_likehood[j][k]!=0:
                    shift_likehood[j][k]=-np.log(shift_likehood[j][k])
                else:
                    shift_likehood[j][k]=np.inf
        self.hmm.transition_cost=np.array(shift_likehood)
    
    def GMMKmeans_WithoutEM(self,nodes_for_Kmeans,num_Gaussian_distribution):
        #initialize with mean, var and weight, with one cluster
        num_templates=len(nodes_for_Kmeans)
        means=[]
        covs=[]
        weights=[1]
        mean=np.mean(nodes_for_Kmeans,axis=0)
        cov=np.diagonal(np.cov(np.array(nodes_for_Kmeans).T),offset=0, axis1=0, axis2=1)
        means.append(mean)
        covs.append(cov)
        
        current_num_of_cluster=1
        episolom=0.04
        #initial should be 1 mean
        mix = mixInfo()
        mix.Gaussian_var = np.array(covs)
        mix.Gaussian_mean = np.array(means)
        mix.Num_of_Gaussian = current_num_of_cluster
        mix.Gaussian_weight = np.array(weights)
        stop=False
        
        while num_Gaussian_distribution>current_num_of_cluster and not stop:
            #now split
            new_means=[]
            new_covs=[]
            current_num_of_cluster=current_num_of_cluster*2
            new_clusters=[]
            for cluster in range(len(means)):
                #append newly two cluster center
                new_clusters.append([])
                new_clusters.append([])
                #get splitted mean and cov
                new_mean1=means[cluster]*(1-episolom)
                new_mean2=means[cluster]*(1+episolom)
                new_cov1=covs[cluster]*(1-episolom)
                new_cov2=covs[cluster]*(1+episolom)
                new_means.append(new_mean1)
                new_means.append(new_mean2)
                new_covs.append(new_cov1)
                new_covs.append(new_cov2)
            #now assign the templated into new clusters
            new_means=np.array(new_means)
            new_covs=np.array(new_covs)
            for node in nodes_for_Kmeans:
                d=log_gaussian(new_means,new_covs,node)
                cluster=np.argmin(d)
                new_clusters[cluster].append(node)
            #now, according to the new clustered result, we get updated weight,
            #mean and cov
            means=[]
            covs=[]
            weights=[]
            #
            print("For {} clusters, each cluster has following nodes".format(current_num_of_cluster))
            for cluster in new_clusters:
                print(len(cluster))
                if len(cluster)<2*num_Gaussian_distribution:
                    stop=True
                    print("For this state, we only have 2 Gaussian Distributions")
                mean=np.mean(cluster,axis=0)
                cov=np.cov(np.array(cluster).T)
                cov=np.diagonal(cov,offset=0, axis1=0, axis2=1)
                weight=len(cluster)/num_templates
                means.append(mean)
                covs.append(cov)
                weights.append(weight)
            #print(np.sum(weights))
            print("get {} means".format(current_num_of_cluster))
            # now, we put all the information to mix
            mix = mixInfo()
            mix.Gaussian_var = np.array(covs)
            mix.Gaussian_mean = np.array(means)
            mix.Num_of_Gaussian = current_num_of_cluster
            mix.Gaussian_weight = np.array(weights)
        return mix
    
    
    def EM(self,nodes,mix):
        #input:
              #nodes are current state the templates for EM
              #curren state mix will be updated
        #EM is to update the information of one state's GMM
        num_guassian_distribution=mix.Num_of_Gaussian
        iteration=0
        while True:
            #aaa=[0,0]
            iteration+=1
            mu=mix.Gaussian_mean
            squared_sigma=mix.Gaussian_var
            alpha=mix.Gaussian_weight
            P_l_X_i_s=[]
            #这边可以利用 numpy 计算优化！！！
            for x_i in nodes:
                all_costs_of_x_i=gaussian(mu,squared_sigma,x_i)  
                weighted_costs_of_x_i=alpha*all_costs_of_x_i
                summed_weighted_cost=np.sum(weighted_costs_of_x_i)
                P_l_X_i_s.append(weighted_costs_of_x_i/summed_weighted_cost)
            #updata alpha: 
            array_P_l_X_i_s=np.array(P_l_X_i_s)
            new_alpha=np.sum(array_P_l_X_i_s,axis=0)/len(P_l_X_i_s)
            #updata mu:
            new_mu=[]
            new_squared_sigma=[]
            array_nodes=np.array(nodes)
            #use np.dot to do the quick sum
            part1=np.dot(array_P_l_X_i_s.T,array_nodes)
            part2=np.sum(P_l_X_i_s,axis=0)
            for l in range(num_guassian_distribution):
                mu_l=part1[l]/part2[l]
                new_mu.append(mu_l)
                #updata squred_cov l:
                part3=array_nodes-mu_l
                p_l=array_P_l_X_i_s[:,l]
                cov_l=np.dot((part3.T*p_l),part3)/part2[l]
                cov_l_diagonal=np.diagonal(cov_l, offset=0, axis1=0, axis2=1)
                new_squared_sigma.append(cov_l_diagonal)
            #https://blog.csdn.net/Elenstone/article/details/105752321
            err=0
            err_alpha=0
            for z in range(num_guassian_distribution):
                err += np.sum(abs(mu[z]-new_mu[z]))      #计算误差
                err_alpha += abs(alpha[z]-new_alpha[z])
#             print(err)
#             print(err_alpha)
#             print()
            #Now, CHR has successfully updata the mean,cov, weight for mix of current_mix
            #let us record this mix
            mix.Gaussian_mean=np.array(new_mu)
            mix.Gaussian_var=np.array(new_squared_sigma)
            mix.Gaussian_weight=np.array(new_alpha)
            if (err<=0.001) and (err_alpha<0.001):     #达到精度退出迭代
                print(err,err_alpha)
                print("Use {} iterations of EM to converge".format(iteration))
                break
        return mix    

    def inithmm(self):
        self.hmm = hmmInfo()
        self.hmm.init = np.zeros((self.state_number,1))
        self.hmm.init[0] = 1
        self.hmm.N = self.state_number
        #update node_in_each_state,node_state
        #initialization of all state is evenly segmented
        self.node_state=[]#each node in which state, all templates together
        for k in range(len(self.templates)):#templates number
            #now, it is the k th training sequence we are look at
            #for initial part, each state have even number of nodes
            n_node=len(self.templates[k])//self.state_number
            num_left_nodes=len(self.templates[k])%self.state_number
            #n_node is the N_kj
            # to store the node state of k th sequence
            current_sample_node_state=np.zeros(len(self.templates[k])).astype(int) 
            #now, initialize the node state for each node in the k th sequence
            for i in range(1,self.state_number+1):
                current_sample_node_state[n_node*(i-1):n_node*i]+=i
            #left nodes be the last state:
            if num_left_nodes!=0:
                current_sample_node_state[-num_left_nodes:]+=self.state_number
            self.node_state.append(current_sample_node_state)
            #to check my initial node state assignment
            #print(np.bincount(self.node_state[k], weights=None, minlength=0))
        #update node in different state
        self.update_node_in_each_state(show_result=True)
        #Markov chain(calculate current edge score)
        self.compute_transition_cost()
        #Now, we have evenly distributed all the mfcc vector to different states
        #next, let us use gaussian to simulate the nodes
        GMMS=[]
        for state in range(self.state_number):
            print("Initilizinng the state {}".format(state+1))
            current_state_nodes=self.node_in_each_state[state+1]
            #kmeans = KMeans(n_clusters = K,random_state=0).fit(np.array(current_state_nodes))
            curren_state_mix=self.GMMKmeans_WithoutEM(current_state_nodes,self.Gaussian_distribution_number[state])
            GMMS.append(curren_state_mix)
        self.hmm.mix=GMMS   
    
    
    def trainhmm(self):
        self.inithmm()
        #complete set up by iteratively update the model vectors,covariance
        #and transition score
        previous_best_distance=-np.inf
        current_best_distance=0
        for j in range(1,100):
            #update the node state
            for k in range(len(self.templates)):
                distance,self.node_state[k]=self.GMM_HMM_dtw(self.templates[k],get_track=True)
                current_best_distance+=distance
            #Once we get the new segment, we updata Markov chain(
            self.compute_transition_cost(show_result=True)
            #according to the new segment, get new node in each state
            self.update_node_in_each_state(show_result=True)
            #according to the updated node state, update hmm
            GMMS=[]
            for state in range(self.state_number):
                print("Update GMM of state {}".format(state+1))
                current_state_nodes=self.node_in_each_state[state+1]
                #kmeans = KMeans(n_clusters = K,random_state=0).fit(np.array(current_state_nodes))
                curren_state_mix=self.GMMKmeans_WithoutEM(current_state_nodes,self.Gaussian_distribution_number[state])
                GMMS.append(curren_state_mix)
            self.hmm.mix=GMMS  
            #Convergence is achieved when the total best-alignment error for
            #all training sequences does not change significantly with further
            #refinement of the model
            difference= previous_best_distance-current_best_distance
    #         print("current difference")
    #         print(difference)
            previous_best_distance=current_best_distance
    #         print("updated previous best distance")
    #         print(previous_best_distance)
            current_best_distance=0
            if abs(difference)<0.0015:
                print("Use {} iterators to updata HMM".format(j))
                break
        #now, since our one Guassian is converged, let me add the transition for the end point, that is:
        #at the end of the state, what is its probability it goes to non_emitting node
        #to do this, we only need to know how many nodes there are at the end of the state 
        #and how many templates we use to train the model
        new_transition=np.zeros((self.state_number+2,self.state_number+2))
        num_nodes_at_last_state=self.state_node_num[self.state_number]
        num_templates=len(self.templates)
        probability_of_get_into_non_emitting_state=num_templates/num_nodes_at_last_state
        log_probability=np.log(probability_of_get_into_non_emitting_state)
        new_transition[:self.state_number+1,:self.state_number+1]=self.hmm.transition_cost
        new_transition[self.state_number+1,self.state_number+1]=log_probability
        self.hmm.transition_cost=new_transition
    
    
    def GMM_HMM_dtw(self,data,get_track=False):
        #Input, T is the transition_score
        #yes, we use cost!!!!!!!!
        #Output: the path, how we align each node
        # insert fin at the beginning of the template and data
        T=self.hmm.transition_cost
        zeros=np.zeros([39])
        ones=np.zeros([39])+1
        mix_of_all_states=[]
        #create a fine GMM 
        fine_GMM=mixInfo()
        fine_GMM.Gaussian_mean.append(zeros)
        #fine_GMM.Gaussian_mean.append(zeros)
        fine_GMM.Gaussian_var.append(ones)
        #fine_GMM.Gaussian_var.append(ones)
        fine_GMM.Gaussian_weight=[1]
        #translate from list to np array
        fine_GMM.Gaussian_mean=np.array(fine_GMM.Gaussian_mean)
        fine_GMM.Gaussian_var=np.array(fine_GMM.Gaussian_var)
        fine_GMM.Num_of_Gaussian = 1
        mix_of_all_states.append(fine_GMM)
        for current_mix in self.hmm.mix:
            mix_of_all_states.append(current_mix)
        data=np.vstack([zeros,data])
        #print(data.shape)

        t=len(mix_of_all_states) # here, t should be the number of states+1 we set
        d=len(data)#means input frame j
        #create empty best path cost matrix "P" 
        P=np.zeros([t,d])
        #to fetch the data, we use P[i][j],i for template and j for input data
    #     • P i,j = best path cost from origin to node [i,j]
    #     • C i,j = local node cost of aligning template frame i to input frame j
    #     • T i,j,k,l = Edge cost from node (i,j) to node (k,l)

        for j in range(0,d): #input frame j
            for i in range(t): # i th template frame aligns with j-th input frame
                #6 PPT p.g. 65
                Cij= mixture_log_gaussian(mix_of_all_states[i],data[j])
                #print(Cij)
                if i-2>=0:
                    P[i][j]=min(P[i][j-1]+T[i][i],P[i-1][j-1]+T[i-1][i],
                                P[i-2][j-1]+T[i-2][i])+Cij
                elif i-1>=0:
                    P[i][j]=min(P[i][j-1]+T[i][i],P[i-1][j-1]+T[i-1][i])+Cij
                else:
                    P[i][j]=P[i][j]+Cij

        #Use DTW cost / frame of input speech, instead of total DTW cost, before determining threshold
        # 5 PPT  p.g 32
        P=P/d
    #     print(P.shape)
    #     print(P[-1][-1])
    #     print(P)
        distance=P[-1][-1]
        if get_track:
            return distance,traceback(P)
        else:
            return distance