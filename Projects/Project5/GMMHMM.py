import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc

def getMFCC2(wavename):  # Compute MFCC features without normalization

    # Read the audio file
    fs, audio = wav.read(wavename)

    # Compute MFCC features
    feature_mfcc = mfcc(audio, samplerate=fs)

    # Initialize list to store modified MFCC features
    mfcc_modified = []

    # Append the first and last MFCC feature vectors three times each
    mfcc_modified.append(np.hstack([feature_mfcc[0], feature_mfcc[0], feature_mfcc[0]]))
    mfcc_modified.append(np.hstack([feature_mfcc[-1], feature_mfcc[-1], feature_mfcc[-1]]))

    # Compute delta features and append to the modified MFCC features
    for i in range(1, len(feature_mfcc) - 1):
        delta = np.zeros(13)
        for j in range(13):
            delta[j] = feature_mfcc[i + 1][j] - feature_mfcc[i - 1][j]
        mfcc_modified.append(np.hstack([feature_mfcc[i], delta]))

    # Compute acceleration features and append to the modified MFCC features
    for i in range(1, len(mfcc_modified) - 1):
        acc = np.zeros(13)
        for j in range(13):
            acc[j] = mfcc_modified[i + 1][13 + j] - mfcc_modified[i - 1][13 + j]
        mfcc_modified[i] = np.hstack([mfcc_modified[i], acc])

    # Convert the modified MFCC features to numpy array
    mfccs = np.array(mfcc_modified)

    # Normalize each feature vector by its variance
    var = np.var(mfccs, axis=1)
    for i in range(len(mfccs)):
        for j in range(39):
            mfccs[i][j] = mfccs[i][j] / var[i]

    return mfccs

class mixInfo:
    """Class to store information about Gaussian mixture models."""
    def __init__(self):
        self.Gaussian_mean = []  # Mean vectors for each Gaussian distribution
        self.Gaussian_var = []   # Diagonal covariance matrices for each Gaussian distribution
        self.Gaussian_weight = []  # Weights for each Gaussian distribution (sums to 1)
        self.Num_of_Gaussian = 0  # Number of Gaussian distributions

class hmmInfo:
    '''Class to store Hidden Markov Model (HMM) parameters.'''
    def __init__(self):
        self.init = []  # Initial matrix
        self.transition_cost = []  # Transition costs between states
        self.mix = []  # Parameters for Gaussian mixture models, each state has its own mix
        self.N = 0  # Number of states

def log_gaussian(mu, squared_sigma, input_vector):
    """
    Compute the cost using log Gaussian.

    Args:
        mu: Mean vectors.
        squared_sigma: Diagonal covariance matrices.
        input_vector: Input vector.

    Returns:
        The cost computed using log Gaussian.
    """
    # Calculate the cost using log Gaussian
    part1 = 0.5 * np.sum(np.log((2 * np.pi) * (squared_sigma)), axis=1)
    part2 = 0.5 * np.sum(np.square((input_vector - mu)) / squared_sigma, axis=1)
    cost = part1 + part2
    return cost

def gaussian(mu, squared_sigma, input_vector):
    """
    Compute the probability density function of a Gaussian distribution.

    Args:
        mu: Mean vectors.
        squared_sigma: Diagonal covariance matrices.
        input_vector: Input vector.

    Returns:
        The probability density function values.
    """
    # Calculate the probability density function
    d = input_vector.shape[0]  # Dimensionality of the input vector
    part0 = np.prod(squared_sigma, axis=1)
    part1 = np.sqrt((2 * np.pi) ** d * part0)
    front = 1 / part1
    part2 = 0.5 * np.sum((mu - input_vector) ** 2 / squared_sigma, axis=1)
    expo = np.exp(-part2)
    p = front * expo
    return p

def mixture_log_gaussian(mix, input_vector):
    """
    Compute the log likelihood of an input vector given a Gaussian mixture model.

    Args:
        mix: Gaussian mixture model parameters.
        input_vector: Input vector.

    Returns:
        The log likelihood of the input vector given the Gaussian mixture model.
    """
    weight = mix.Gaussian_weight
    mu = mix.Gaussian_mean
    squared_sigma = mix.Gaussian_var
    cost = log_gaussian(mu, squared_sigma, input_vector)
    weighted_cost = np.sum(weight * cost)
    return weighted_cost

def traceback(D):
    """
    Perform traceback to find the most likely state sequence.

    Args:
        D: Matrix containing the accumulated costs.

    Returns:
        The most likely state sequence.
    """
    # Start from the last state and last frame
    current_state, current_frame = np.array(D.shape) - 1
    # Insert the last frame's state
    x = [current_state]

    # Perform traceback until reaching the first frame
    while current_state > 0 and current_frame > 1:
        # Move to the previous frame
        current_frame -= 1

        # Determine the most likely transition
        if current_state > 2:
            to_check = [D[current_state][current_frame - 1],
                        D[current_state - 1][current_frame - 1],
                        D[current_state - 2][current_frame - 1]]
            track = np.argmin(to_check)
            if track == 2:
                print(to_check)
                print(current_frame)
        elif current_state > 1:
            to_check = [D[current_state][current_frame - 1],
                        D[current_state - 1][current_frame - 1]]
            track = np.argmin(to_check)
        else:
            track = 0

        # Update the current state based on the most likely transition
        if track == 0:
            # Last frame remains in the same state
            x.insert(0, current_state)
        elif track == 1:
            current_state -= 1
            x.insert(0, current_state)
        else:
            current_state -= 2
            x.insert(0, current_state)

    return x

class GMMHMM(object):

    def __init__(self, templates, Gaussian_distribution_number=[4,4,4,4,4]):
        """
        Initialize the object with templates and Gaussian distribution numbers.

        Args:
            templates: The templates for the specific word.
            Gaussian_distribution_number: The number of Gaussian distributions for each state.

        Returns:
            None
        """
        self.templates = templates
        # Length should be state_number
        self.Gaussian_distribution_number = Gaussian_distribution_number
        self.state_number = len(self.Gaussian_distribution_number)
        self.node_in_each_state = []
        self.node_state = []
        self.hmm = None

    def update_node_in_each_state(self, show_result=False):
        """
        Update the nodes in each state based on the current node state assignment.

        Args:
            show_result: A boolean indicating whether to print the updated number of nodes in each state.

        Returns:
            None
        """
        self.node_in_each_state = []  # State number decides sublist number
        # self.node_in_each_state[0] is empty
        for state in range(self.state_number + 1):
            self.node_in_each_state.append([])

        for k in range(len(self.templates)):  # Templates number
            # The i-th vector of the k-th training sequence
            for i in range(len(self.node_state[k])):
                j = int(self.node_state[k][i])  # The state of the i-th vector
                self.node_in_each_state[j].append(self.templates[k][i])

    def compute_transition_cost(self, show_result=False):
        """
        Compute the transition cost matrix.

        Args:
            show_result: A boolean indicating whether to print the number of nodes in different states.

        Returns:
            None
        """
        shift_likelihood = np.zeros((self.state_number + 1, self.state_number + 1))
        self.state_node_num = np.zeros(self.state_number + 1)
        initial_states = []

        # Count the state transition of all the nodes
        for k in range(len(self.node_state)):
            for i in range(len(self.node_state[k]) - 1):
                current_node = self.node_state[k][i]
                next_node = self.node_state[k][i + 1]
                shift_likelihood[current_node][next_node] += 1
                self.state_node_num[current_node] += 1
            # Last node case
            shift_likelihood[self.node_state[k][-2]][self.node_state[k][-1]] += 1
            self.state_node_num[self.node_state[k][-1]] += 1

        if show_result:
            print("The num of nodes in different states are {}".format(self.state_node_num))

        # Compute transition probabilities and convert them to transition costs
        for j in range(self.state_number + 1):
            N = len(self.node_state)
            N_0j = shift_likelihood[0][j]
            shift_likelihood[0][j] = N_0j / N
            if N_0j == 0:
                shift_likelihood[0][j] = np.inf
            else:
                shift_likelihood[0][j] = -np.log(shift_likelihood[0][j])

        for j in range(1, self.state_number + 1):
            for k in range(j, self.state_number + 1):
                shift_likelihood[j][k] = shift_likelihood[j][k] / self.state_node_num[j]
                if shift_likelihood[j][k] != 0:
                    shift_likelihood[j][k] = -np.log(shift_likelihood[j][k])
                else:
                    shift_likelihood[j][k] = np.inf

        self.hmm.transition_cost = np.array(shift_likelihood)

    def GMMKmeans_WithoutEM(self, nodes_for_Kmeans, num_Gaussian_distribution):
        """
        Perform K-means clustering to initialize Gaussian Mixture Model parameters without EM algorithm.

        Args:
            nodes_for_Kmeans: A list of data points used for K-means clustering.
            num_Gaussian_distribution: Number of Gaussian distributions desired in the GMM.

        Returns:
            mix: An instance of mixInfo containing the initialized Gaussian Mixture Model parameters.
        """
        # Initialize with mean, variance, and weight for one cluster
        num_templates = len(nodes_for_Kmeans)
        means = []
        covs = []
        weights = [1]
        mean = np.mean(nodes_for_Kmeans, axis=0)
        cov = np.diagonal(np.cov(np.array(nodes_for_Kmeans).T), offset=0, axis1=0, axis2=1)
        means.append(mean)
        covs.append(cov)
        current_num_of_cluster = 1
        episolom = 0.04
        mix = mixInfo()  # Initialize mixInfo object to store GMM parameters
        mix.Gaussian_var = np.array(covs)
        mix.Gaussian_mean = np.array(means)
        mix.Num_of_Gaussian = current_num_of_cluster
        mix.Gaussian_weight = np.array(weights)
        stop = False
        
        # Iterate until desired number of Gaussian distributions is reached or stop condition is met
        while num_Gaussian_distribution > current_num_of_cluster and not stop:
            new_means = []
            new_covs = []
            current_num_of_cluster = current_num_of_cluster * 2
            new_clusters = []
            
            # Split existing clusters and assign data points to new clusters
            for cluster in range(len(means)):
                new_clusters.append([])
                new_clusters.append([])
                new_mean1 = means[cluster] * (1 - episolom)
                new_mean2 = means[cluster] * (1 + episolom)
                new_cov1 = covs[cluster] * (1 - episolom)
                new_cov2 = covs[cluster] * (1 + episolom)
                new_means.append(new_mean1)
                new_means.append(new_mean2)
                new_covs.append(new_cov1)
                new_covs.append(new_cov2)
            
            new_means = np.array(new_means)
            new_covs = np.array(new_covs)
            
            # Assign data points to new clusters
            for node in nodes_for_Kmeans:
                d = log_gaussian(new_means, new_covs, node)
                cluster = np.argmin(d)
                new_clusters[cluster].append(node)
            
            # Update means, covariances, and weights based on new clustering
            means = []
            covs = []
            weights = []
            print("For {} clusters, each cluster has following nodes".format(current_num_of_cluster))
            
            for cluster in new_clusters:
                print(len(cluster))
                if len(cluster) < 2 * num_Gaussian_distribution:
                    stop = True
                    print("For this state, we only have 2 Gaussian Distributions")
                mean = np.mean(cluster, axis=0)
                cov = np.cov(np.array(cluster).T)
                cov = np.diagonal(cov, offset=0, axis1=0, axis2=1)
                weight = len(cluster) / num_templates
                means.append(mean)
                covs.append(cov)
                weights.append(weight)
            
            print("get {} means".format(current_num_of_cluster))
            
            # Update mixInfo object with new GMM parameters
            mix = mixInfo()
            mix.Gaussian_var = np.array(covs)
            mix.Gaussian_mean = np.array(means)
            mix.Num_of_Gaussian = current_num_of_cluster
            mix.Gaussian_weight = np.array(weights)
        
        return mix

    def EM(self, nodes, mix):
        """
        Update the parameters of a Gaussian Mixture Model (GMM) using the Expectation-Maximization (EM) algorithm.

        Args:
            nodes: The data points for the current state.
            mix: An instance of mixInfo containing the current GMM parameters.

        Returns:
            mix: An instance of mixInfo containing the updated GMM parameters.
        """
        num_guassian_distribution = mix.Num_of_Gaussian
        iteration = 0
        
        while True:
            iteration += 1
            mu = mix.Gaussian_mean
            squared_sigma = mix.Gaussian_var
            alpha = mix.Gaussian_weight
            P_l_X_i_s = []
            
            # E-step: Calculate the probability of each data point belonging to each Gaussian distribution
            for x_i in nodes:
                all_costs_of_x_i = gaussian(mu, squared_sigma, x_i)  
                weighted_costs_of_x_i = alpha * all_costs_of_x_i
                summed_weighted_cost = np.sum(weighted_costs_of_x_i)
                P_l_X_i_s.append(weighted_costs_of_x_i / summed_weighted_cost)
            
            # M-step: Update parameters
            array_P_l_X_i_s = np.array(P_l_X_i_s)
            
            # Update alpha
            new_alpha = np.sum(array_P_l_X_i_s, axis=0) / len(P_l_X_i_s)
            
            # Update mu and covariance
            new_mu = []
            new_squared_sigma = []
            array_nodes = np.array(nodes)
            part1 = np.dot(array_P_l_X_i_s.T, array_nodes)
            part2 = np.sum(P_l_X_i_s, axis=0)
            
            for l in range(num_guassian_distribution):
                mu_l = part1[l] / part2[l]
                new_mu.append(mu_l)
                
                # Update covariance
                part3 = array_nodes - mu_l
                p_l = array_P_l_X_i_s[:, l]
                cov_l = np.dot((part3.T * p_l), part3) / part2[l]
                cov_l_diagonal = np.diagonal(cov_l, offset=0, axis1=0, axis2=1)
                new_squared_sigma.append(cov_l_diagonal)
            
            # Calculate error for convergence check
            err = 0
            err_alpha = 0
            
            for z in range(num_guassian_distribution):
                err += np.sum(abs(mu[z] - new_mu[z]))
                err_alpha += abs(alpha[z] - new_alpha[z])
            
            # Update mixInfo object with new parameters
            mix.Gaussian_mean = np.array(new_mu)
            mix.Gaussian_var = np.array(new_squared_sigma)
            mix.Gaussian_weight = np.array(new_alpha)
            
            # Check for convergence
            if (err <= 0.001) and (err_alpha < 0.001):
                print(err, err_alpha)
                print("Use {} iterations of EM to converge".format(iteration))
                break
        
        return mix

    def inithmm(self):
        """
        Initialize the Hidden Markov Model (HMM) parameters.

        Returns:
            None
        """
        # Initialize the HMM object
        self.hmm = hmmInfo()
        self.hmm.init = np.zeros((self.state_number, 1))
        self.hmm.init[0] = 1
        self.hmm.N = self.state_number

        # Update node_in_each_state and node_state
        self.node_state = []
        for k in range(len(self.templates)):
            n_node = len(self.templates[k]) // self.state_number
            num_left_nodes = len(self.templates[k]) % self.state_number
            current_sample_node_state = np.zeros(len(self.templates[k])).astype(int)
            for i in range(1, self.state_number + 1):
                current_sample_node_state[n_node * (i - 1):n_node * i] += i
            if num_left_nodes != 0:
                current_sample_node_state[-num_left_nodes:] += self.state_number
            self.node_state.append(current_sample_node_state)

        self.update_node_in_each_state(show_result=True)

        # Calculate transition costs
        self.compute_transition_cost()

        # Initialize Gaussian Mixture Models (GMMs)
        GMMS = []
        for state in range(self.state_number):
            print("Initializing state {}".format(state + 1))
            current_state_nodes = self.node_in_each_state[state + 1]
            current_state_mix = self.GMMKmeans_WithoutEM(current_state_nodes, self.Gaussian_distribution_number[state])
            GMMS.append(current_state_mix)

        self.hmm.mix = GMMS

    def trainhmm(self):
        """
        Train the Hidden Markov Model (HMM).

        Returns:
            None
        """
        self.inithmm()

        previous_best_distance = -np.inf
        current_best_distance = 0

        for j in range(1, 100):
            # Update node state
            for k in range(len(self.templates)):
                distance, self.node_state[k] = self.GMM_HMM_dtw(self.templates[k], get_track=True)
                current_best_distance += distance

            # Update Markov chain
            self.compute_transition_cost(show_result=True)

            # Update node in each state
            self.update_node_in_each_state(show_result=True)

            # Update HMM parameters
            GMMS = []
            for state in range(self.state_number):
                print("Updating GMM of state {}".format(state + 1))
                current_state_nodes = self.node_in_each_state[state + 1]
                current_state_mix = self.GMMKmeans_WithoutEM(current_state_nodes, self.Gaussian_distribution_number[state])
                GMMS.append(current_state_mix)
            self.hmm.mix = GMMS

            # Check convergence
            difference = previous_best_distance - current_best_distance
            previous_best_distance = current_best_distance
            current_best_distance = 0

            if abs(difference) < 0.0015:
                print("Used {} iterations to update HMM".format(j))
                break

        # Update transition for the end point
        new_transition = np.zeros((self.state_number + 2, self.state_number + 2))
        num_nodes_at_last_state = self.state_node_num[self.state_number]
        num_templates = len(self.templates)
        probability_of_get_into_non_emitting_state = num_templates / num_nodes_at_last_state
        log_probability = np.log(probability_of_get_into_non_emitting_state)
        new_transition[:self.state_number + 1, :self.state_number + 1] = self.hmm.transition_cost
        new_transition[self.state_number + 1, self.state_number + 1] = log_probability
        self.hmm.transition_cost = new_transition

    def trainhmm(self):
        """
        Train the Hidden Markov Model (HMM) parameters.
        
        Returns:
            None
        """
        # Initialize HMM parameters
        self.inithmm()

        # Iteratively update the model vectors, covariance, and transition scores
        previous_best_distance = -np.inf
        current_best_distance = 0
        
        # Iterate for a maximum of 100 times (can be adjusted)
        for j in range(1, 100):
            # Update the node state and calculate the best alignment distance
            for k in range(len(self.templates)):
                distance, self.node_state[k] = self.GMM_HMM_dtw(self.templates[k], get_track=True)
                current_best_distance += distance
            
            # Update transition costs based on the new node state
            self.compute_transition_cost(show_result=True)
            
            # Update the node distribution in each state
            self.update_node_in_each_state(show_result=True)
            
            # Update the Gaussian Mixture Models (GMMs) for each state
            GMMS = []
            for state in range(self.state_number):
                print("Updating GMM of state {}".format(state + 1))
                current_state_nodes = self.node_in_each_state[state + 1]
                current_state_mix = self.GMMKmeans_WithoutEM(current_state_nodes, self.Gaussian_distribution_number[state])
                GMMS.append(current_state_mix)
            self.hmm.mix = GMMS  

            # Check for convergence
            difference = previous_best_distance - current_best_distance
            previous_best_distance = current_best_distance
            current_best_distance = 0
            
            if abs(difference) < 0.0015:
                print("Used {} iterations to update HMM".format(j))
                break

        # Update the transition for the end point
        self.update_end_transition()
