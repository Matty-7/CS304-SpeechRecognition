from typing import Dict, Optional, List
import numpy as np
import math

class HMMState:
    def __init__(self, mean=None, covariance=None, label1= None,isNull=False):
        self.mean = mean
        """n_gaussians of mean vectors."""
        self.covariance = covariance
        """n_gaussians of diagonal of covariance matrix."""
        self.label1 = label1
        """The digit associated with the state. `None` if the state is the first state."""
        self.parents = []
        """The state is the first state if the `parent` is `None`."""
        self.isNull=isNull
    
    def log_multivariate_gaussian_pdf_diag_cov(self, x, epsilon=1e-9):
        mean = self.mean
        d = x.shape[0]
    
    # Regularize the covariance matrix by adding epsilon to its diagonal
        cov_safe = self.covariance + epsilon * np.eye(d)
    
    # Calculate the log determinant and the inverse of the regularized covariance matrix
        log_det_cov = np.log(np.linalg.det(cov_safe))
        inv_cov = np.linalg.inv(cov_safe)
    
    # Compute the constant term of the Gaussian PDF
        const_term = -0.5 * d * np.log(2 * np.pi)
    
    # Compute the quadratic term
        diff = x - mean
        quadratic_term = -0.5 * np.dot(diff.T, np.dot(inv_cov, diff))
    
    # Compute the log PDF
        log_pdf = const_term - 0.5 * log_det_cov + quadratic_term
        return log_pdf
        
    def get_log_emission_prob(self, observation: np.ndarray) -> float:
     
        if not self.mean or not self.covariance:
            return -np.inf  

        log_probs = [
            log_multivariate_gaussian_pdf_diag_cov(observation, mean, cov)
            for mean, cov in zip(self.mean, self.covariance)
        ]
        
      
        max_log_prob = max(log_probs)
        
       
        log_sum = np.log(np.sum(np.exp(log_probs - max_log_prob))) + max_log_prob
        
        return log_sum - np.log(len(self.mean)) 


def hmm_load_features(data_dir):
    #创建一个空列表来存储样本对象
    samples = []

    #*循环读取数据并创建样本对象，每个样本对象包括特征数据和标签
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.npy'):
            # 解析文件名，获取标签信息
            parts = file_name.split('-')
            if len(parts) == 2 and parts[1].endswith('.npy'):
                label = int(parts[0])
                
                # 加载特征数据
                features = np.load(os.path.join(data_dir, file_name))

                # 创建样本对象并添加到列表中
                sample = {'label': label, 'features': features}
                samples.append(sample)

    return samples

def filter_samples_by_label(samples, label):
    """
    Filters the list of sample dictionaries to include only those with a specific label.

    :param samples: List of dictionaries, where each dictionary contains 'label' and 'features' keys.
    :param label: The label to filter by (default is 1).
    :return: A filtered list of dictionaries.
    """
    return [sample["features"] for sample in samples if sample['label'] == label]
    # Assuming the training folder path is correct

from typing import List, Dict,Tuple
import numpy as np
import os
from sklearn.cluster import KMeans


class HMM:
    def __init__(self,label,training_folder_path='training'):
        self.states: List[HMMState] = []
        data=hmm_load_features(training_folder_path)
        sequences = [sample['features'] for sample in data]
    
        templates_for_label=filter_samples_by_label(data, label)
        self.templates=templates_for_label
        self.num_states=self.get_num_state()
        self.observations: List[np.ndarray] = []
        self.transitions: List[List[float]] = []
        self.state_index: Dict[HMMState, int] = {}
        self.initial_probabilities: List[float] = []  # Probability of starting in each state
    def calculate_mean_and_covariance(self,vectors):
    
    # Convert the list of vectors to a NumPy array for easier calculations
        vectors_np = np.array(vectors)
    
    # Calculate the mean vector
        mean_vector = np.mean(vectors_np, axis=0)
    
    # Calculate the variance vector
        covariance_matrix = np.cov(vectors_np.T)
    
        return mean_vector, covariance_matrix
    
    def normalize_sequence(self,seq):
        if not seq:
            return seq  # Return empty list if input is empty

        normalized_seq = [seq[0]]  # Start with the first element

        for i in range(1, len(seq)):
            current = seq[i]
            previous = normalized_seq[-1]

        # If current continues the trend or equals the previous, it's normal
            if current >= previous:
                normalized_seq.append(current)
            else:
            # Look ahead to see if this is a temporary dip or start of a new trend
                if i + 1 < len(seq) and seq[i + 1] >= current:
                # If next is greater than or equal to current, current is abnormal; repeat previous
                    normalized_seq.append(previous)
                else:
                # Otherwise, start of a new trend or continuation of a decrease
                    normalized_seq.append(current)

        return normalized_seq
    def print_status(self):
        print("HMM Status Report")
        print("=================")
        print(f"Number of States: {len(self.states)}")
        print(f'state index: {self.state_index}')
        # Optionally, print details about each state if HMMState has identifiable attributes
        for i, state in enumerate(self.states):
            print(f"  State {i}: {state}")  # Customize based on HMMState's attributes

        print(f"Number of Observations: {len(self.observations)}")
        # Optionally, print details about observations if they're simple enough to summarize
        for i, obs in enumerate(self.observations):
            print(f"  Observation {i}: Shape {obs.shape}")

        print(f"Transition Matrix: {len(self.transitions)}x{len(self.transitions)}" if self.transitions else "Not defined")
        for i, row in enumerate(self.transitions):
            print(f"  Transition from State {i}: {row}")

        print(f"State Index Map: {len(self.state_index)} entries")
        for state, index in self.state_index.items():
            print(f"  State {state} -> Index {index}")  # Customize based on HMMState's attributes

        print(f"Initial Probabilities: {self.initial_probabilities}")

    def add_state(self, state: HMMState):
        """Adds a state to the HMM."""
        self.states.append(state)
        
        # Ensure transitions matrix is updated to reflect the new state
        for row in self.transitions:
            row.append(0.0)  # Append 0.0 for new state to existing states
        self.transitions.append([0.0 for _ in range(len(self.states))])  # Add new state with transitions
    def initialize_HMM_states(self,label,training_folder_path = 'training'):
          # This path will need to be updated to the actual path
        data=hmm_load_features(training_folder_path)
        sequences = [sample['features'] for sample in data]
    
        templates_for_label=filter_samples_by_label(data, label)
#a is the list of all the templates for digit=label,len(a)=num of templates, len(a[0])=num of segments,len(a[0][0])=length of the first segment, len(a[0][0][0])=39, which is the dimension of the mfcc vector
        a=self.initial_segmentation(templates_for_label,5)
        #print(f'splited all the {len(a)} templates for digit {label} into {len(a[0])} segments uniformly')
        self.get_clusters(a)
        mean=[]
        cov=[]
        for i in range(5):
            m,cv=self.calculate_mean_and_covariance(self.get_clusters(a)[i])
            mean.append(m)
            cov.append(cv)
        return mean,cov
    #def calculate_transition_probabilities(self):
        #for
    

    def initialize(self,label,training_folder_path = 'training',num_states = 5):
        data=hmm_load_features(training_folder_path)
        sequences = [sample['features'] for sample in data]
    
        templates_for_label=filter_samples_by_label(data, label)
#a is the list of all the templates for digit=label,len(a)=num of templates, len(a[0])=num of segments,len(a[0][0])=length of the first segment, len(a[0][0][0])=39, which is the dimension of the mfcc vector
        a=self.initial_segmentation(templates_for_label,5)
        #print(f'len(a):{len(a)}')
        #print(f'len(a[0]):{len(a[0])}')
        clustered_data=self.get_clusters(a)
        mean,var=self.initialize_HMM_states(label)
        for i in range(num_states):
            # Create a new state and add it
            new_state = HMMState(mean[i],var[i],label1=label)
            self.state_index[new_state]=i
            self.add_state(new_state)
        
        # Set initial probabilities (uniform distribution for simplicity)
        self.initial_probabilities = [1 if i==0 else 0 for i in range(num_states)]
        
        # Set up transitions
        for i in range(num_states):
            if i < num_states - 1:
                self.transitions[i][i + 1] = len(templates_for_label)/len(clustered_data[i])
                self.transitions[i][i] = 1- self.transitions[i][i + 1] # Probability of staying in the same state
                  # Probability of moving to the next state
            else:
                self.transitions[i][i] = 1.0  # Last state only points to itself


    def set_observations(self, observations: List[np.ndarray]):
        """Sets the sequence of observations for the HMM."""
        self.observations = observations

    def most_probable_sequence(self, obs_seq):
        V = [{}]
        path = {}

        # Initialize base cases (t == 0)
        for state in self.states:
            initial_prob = self.initial_probabilities[self.state_index[state]]
            V[0][self.state_index[state]] = (math.log(initial_prob) if initial_prob > 0 else -math.inf) + state.log_multivariate_gaussian_pdf_diag_cov(obs_seq[0])
            path[self.state_index[state]] = [state]

        # Run Viterbi for t > 0
        for t in range(1, len(obs_seq)):
            V.append({})
            newpath = {}
            for cur_state in self.states:
                max_log_prob = -math.inf  # Initialize with negative infinity for comparison
                best_prev_state = None  # Initialize with None to find the best previous state
                for prev_state in self.states:
                    transition_prob = self.transitions[self.state_index[prev_state]][self.state_index[cur_state]]
                    log_transition_prob = math.log(transition_prob) if transition_prob > 0 else -math.inf
                    log_prob = V[t-1][self.state_index[prev_state]] + log_transition_prob + cur_state.log_multivariate_gaussian_pdf_diag_cov(obs_seq[t])
                    if log_prob > max_log_prob:
                        max_log_prob = log_prob
                        best_prev_state = prev_state
                V[t][self.state_index[cur_state]] = max_log_prob
                if best_prev_state is not None:  # Check to ensure there is a valid previous state
                    newpath[self.state_index[cur_state]] = path[self.state_index[best_prev_state]] + [cur_state]
            path = newpath

        # Find the final state with the highest probability
        max_final_log_prob = max(V[-1].values())
        final_state = [state for state, prob in V[-1].items() if prob == max_final_log_prob][0]

        return (max_final_log_prob, path[final_state])

    

    def initial_segmentation(self, templates, num_segments):
    
        segmented_templates = []

        for template in templates:
        # Determine the size of each segment
            num_observations = len(template)
            segment_size = num_observations // num_segments
            extra = num_observations % num_segments

            segments = []
            start_idx = 0

            for _ in range(num_segments):
            # Adjust segment size to distribute remaining observations
                end_idx = start_idx + segment_size + (1 if extra > 0 else 0)
            # Decrease extra count until it's distributed
                extra -= 1 if extra > 0 else 0

            # Extract the segment and add to the list
                segment = template[start_idx:end_idx]
                segments.append(segment)

                start_idx = end_idx

            segmented_templates.append(segments)

        return segmented_templates
    def get_clusters(self,segmented_templates,num_segments=5):
        a={}
        
        for i in range(len(segmented_templates)):
            for j in range(len(segmented_templates[i])):
                if i==0:
                    a[j]=np.array(segmented_templates[i][j])
                else:
                    if j>=num_segments:
                        continue
                    else:
                        if j not in a.keys():
                            a[j]=np.array(segmented_templates[i][j])
                        else:
                            a[j]=np.concatenate((a[j],np.array(segmented_templates[i][j])))
        
        return a
    def segment_based_on_indices(self,template,indices):
        segmented_template=[]
        if len(indices)!=0:
            segmented_template.append(template[:indices[0]])
            for i in range(len(indices)-1):
                segment=template[indices[i]:indices[i+1]]
                segmented_template.append(segment)
            if indices[len(indices)-1]!=len(template):
                segmented_template.append(template[indices[len(indices)-1]:])
            else:
                segmented_template.append(template[indices[len(indices)-1]-1:])

        return segmented_template
    def get_num_state(self):
        return len(self.states)


    def train_single_iteration(self):
        
        templates=self.templates
        segmented_templates=[]
        split_indices=[]
        score_total=[]
        for i in range(len(templates)):
            compare_template=templates[i]
            (p,s)=self.most_probable_sequence(compare_template)
            score_total.append(p)
            n=self.normalize_sequence([self.state_index[i] for i in s])
            
            indices=[i for i in range(len(n)-1) if n[i]!=n[i+1]]
            split_indices.append(indices)
            
            segmented_template=self.segment_based_on_indices(compare_template,indices)
            segmented_templates.append(segmented_template)

        score=np.sum(score_total)
        #print(f'len(segmented_templates): {len(segmented_templates)}')
        #print(f'len(segmented_templates[0]):{min(len(segmented_templates[i]) for i in range(10))}')
        clusted_data=self.get_clusters(segmented_templates)
        mean=[]
        cov=[]
        #update emission probabilities
        for i in range(5):
            m,cv=self.calculate_mean_and_covariance(clusted_data[i])
            mean.append(m)
            cov.append(cv)
        #print(mean)
        for i in range(len(self.states)):
            self.states[i].mean=mean[i]
            self.states[i].covariance=cov[i]
        #update transition probabilities
        
        for i in range(len(self.states)):
            if i < len(self.states) - 1:
                #print(f'len(templates): {len(templates)}')
                #print(f'len(clusted_data[i]): {len(clusted_data[i])}')
                self.transitions[i][i + 1] = len(templates)/len(clusted_data[i])
                self.transitions[i][i] = 1- self.transitions[i][i + 1] # Probability of staying in the same state
        # Probability of moving to the next state
            else:
                self.transitions[i][i] = 1.0 
        return score
    def train(self, iterations=10):
        for i in range(iterations):
            s=self.train_single_iteration()
            print(f'HMM training for the {i}th iteration, training score: {s}')
    def evaluate(self, sequences, labels):
        """
        Evaluate the HMM on a test set.
        Args:
            sequences (List[List[np.ndarray]]): A list of observation sequences.
            labels (List[List[int]]): The true state sequences for each observation sequence.
        Returns:
            float, float: The sentence accuracy and the word accuracy.
        """
        correct_sentences = 0
        correct_words = 0
        total_words = 0

        for obs_seq, true_states in zip(sequences, labels):
            predicted_states = self.decode(obs_seq)

            if predicted_states == true_states:
                correct_sentences += 1

            correct_words += sum(p == t for p, t in zip(predicted_states, true_states))
            total_words += len(true_states)

        sentence_accuracy = correct_sentences / len(sequences)
        word_accuracy = correct_words / total_words

        return sentence_accuracy, word_accuracy

import pickle

def save_hmm(hmm, filename):
    """
    Save a trained Hidden Markov Model (HMM) to a file using pickle.

    Parameters:
    - hmm: The HMM object to save.
    - filename: The name of the file where the HMM should be saved.
    """
    with open(filename, 'wb') as file:
        pickle.dump(hmm, file)
    print(f"HMM model has been saved to '{filename}'")
def load_hmm(filename):
    """
    Load a trained Hidden Markov Model (HMM) from a file using pickle.

    Parameters:
    - filename: The name of the file from which to load the HMM.

    Returns:
    - The loaded HMM object.
    """
    with open(filename, 'rb') as file:
        hmm = pickle.load(file)
    print(f"HMM model has been loaded from '{filename}'")
    return hmm

def train_all_HMM(iterations=20):
    all_label=range(1,10)
    for i in all_label:
        filename=f'Digit {i} HMM'
        print(f"Training {filename}")
        hmm=HMM(label=i)
        hmm.initialize_HMM_states(label=i)
        hmm.initialize(label=i)
        hmm.train(iterations)
        save_hmm(hmm, filename)
        print(f'{filename} training finished! Moving to the next.')
        
   
def load_all_hmm():
    hmm1=[]
    for i in range(10):
        hmm=load_hmm(f'Digit {i} HMM')
        hmm1.append(hmm)
    return hmm1

def recognize(hmm1,data,digit):
    p_max=-math.inf
    for i in range(10):
        
        p,s=hmm1[i].most_probable_sequence(data)
        if p>p_max:
            p_max=p
            j=i
    print(f"The voice is recognized as {j}, the true value is {digit}")
    if j==digit:
        print("Congrats, you recognized digit right")
    else:
        print("Opps, it seems that you are wrong")
import librosa
def compute_mfcc_features(file_path, n_mfcc=39):
    y, sr = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    return features

def process_folder(folder_path):
    features_dict = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):  # Ensure processing only wav files
            file_path = os.path.join(folder_path, file_name)
            features = compute_mfcc_features(file_path)
            features_dict[file_name] = features
    return features_dict