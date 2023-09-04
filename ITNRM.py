"""
This script performs "Implicit-Trust-Network-based Recommendation Methodology" (ITNRM) developed by Reza Barzegar Nozari and Hamidreza Koohi [1]. 
It uses user-item ratings data and computes trust values among users to improve the accuracy of 
rating predictions. The script involves several steps, including data preparation, trust calculation, 
trust propagation, and evaluation.

Requirements:
- numpy
- pandas
- numba
- scipy
- sklearn

Usage:
- Ensure you have the required libraries installed.
- Prepare a CSV file named 'Data.csv' containing user-item ratings data with columns 'userId', 'movieId', and 
  'rating'.
- Run the script.

Note: Make sure to adjust the 'Metrics Threshold' according to your dataset's rating scale.

Author: Reza Barzegar Nozari
Date: [Current Date]
"""


import numpy as np
import pandas as pd
import numba as nb
from numba import jit, prange
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split, KFold
import math
import time


# Create user-item data frame
def create_user_item_df(df):
    """
    Creates a user-item matrix from a given DataFrame.

    Parameters:
    df (DataFrame): Input DataFrame containing 'userId', 'movieId', and 'rating' columns.

    Returns:
    DataFrame: User-item matrix where rows represent users, columns represent movies, and values represent ratings.
    """
    
    user_item_df = df.pivot(index='userId', columns='movieId', values='rating')
    
    return user_item_df.fillna(0)


# Main function to compute distance data frame
def calculate_distance(user_item_matrix):
    """
    Computes the distance matrix between users based on their rating vectors.

    Parameters:
    user_item_matrix (ndarray): User-item matrix containing ratings.

    Returns:
    ndarray: Distance matrix between users.
    """
    
    # Precompute the necessary data for indexing pairs of users
    def precompute_data(user_ids):
        num_combinations = len(user_ids) * (len(user_ids) - 1) // 2
        data = np.zeros(num_combinations)
        row_indices = np.zeros(num_combinations, dtype=np.int32)
        col_indices = np.zeros(num_combinations, dtype=np.int32)

        index = 0
        for i in range(len(user_ids) - 1):
            for j in range(i + 1, len(user_ids)):
                row_indices[index] = i
                col_indices[index] = j
                index += 1

        return data, row_indices, col_indices

    # Numba-compiled function to calculate distances
    @jit(nopython=True, parallel=True)
    def calculate_distances(data, row_indices, col_indices, user_item_matrix):
        for i in prange(len(data)):
            distance = data[i]
            u = row_indices[i]
            v = col_indices[i]
            
            u_rating_vector = user_item_matrix[u]
            v_rating_vector = user_item_matrix[v]
            
            intersected_items_of_uv = np.intersect1d(np.where(u_rating_vector > 0)[0], np.where(v_rating_vector > 0)[0])
            if len(intersected_items_of_uv) > 0:
                u_ratings = u_rating_vector[intersected_items_of_uv]
                v_ratings = v_rating_vector[intersected_items_of_uv]
                
                squared_diff_sum = np.sum((u_ratings - v_ratings) ** 2)
                distance = np.sqrt(squared_diff_sum)
                
                data[i] = distance
                
        return data


    data, row_indices, col_indices = precompute_data(user_ids)
    
    # Calculate distances using Numba-compiled function
    calculate_distances(data, row_indices, col_indices, user_item_matrix)

    # Create a sparse distance matrix
    Distance_sparse = csr_matrix((data, (row_indices, col_indices)), shape=(len(user_ids), len(user_ids)))
    Distance_df = pd.DataFrame(index=user_ids, columns=user_ids, data=0.0)
    Distance_df.values[row_indices, col_indices] = Distance_sparse.data
    Distance_df.values[col_indices, row_indices] = Distance_sparse.data
    
    # Normalize the distance values and compute similarity
    u_sum_dist = Distance_df.sum(axis=0)
    Distance_df = 1 - ((Distance_df / u_sum_dist) ** 0.15)
    
    return Distance_df.to_numpy()


# Main function to calculate similarity matrix
def calculate_similarity(user_item_matrix):
    """
    Calculates the similarity matrix between users based on ratings and trust values.

    Parameters:
    user_item_matrix (ndarray): User-item matrix containing ratings.

    Returns:
    dict: Dictionary containing similarity matrix and threshold values.
    """
    
    def precompute_metrics(user_item_matrix):
        # Calculate Pearson Correlation Coefficient (PCC) matrix
        PCC_matrix = np.corrcoef(user_item_matrix)

        # Calculate a distance matrix using a function called calculate_distance
        Distance_matrix = calculate_distance(user_item_matrix)

        return PCC_matrix, Distance_matrix

    # Calculate precomputed metrics: PCC matrix and Distance matrix
    PCC_matrix, Distance_matrix = precompute_metrics(user_item_matrix)

    # Calculate Similarity matrix using array operations
    numerator = 2 * PCC_matrix * Distance_matrix
    denominator = PCC_matrix + Distance_matrix
    # Calculate similarity_matrix, avoiding division by zero with np.where
    similarity_matrix = np.where(PCC_matrix > 0, numerator / denominator, 0)

    # Calculate the similarity threshold for each user based on similarity values
    positive_count_similarity = np.count_nonzero(similarity_matrix, axis=0)
    sim_threshold = np.sum(similarity_matrix, axis=0) / positive_count_similarity

    # Create a dictionary containing the calculated similarity matrix and threshold
    similarity_dict = {'similarity': similarity_matrix, 'threshold': sim_threshold}

    return similarity_dict


# Main function to calculate confidence matrix
def calculate_confidence(user_item_matrix, similarity_dict):
    """
    Calculates the confidence matrix between users based on similarity and rating ratios.

    Parameters:
    user_item_matrix (ndarray): User-item matrix containing ratings.
    similarity_dict (dict): Dictionary containing similarity matrix and threshold values.

    Returns:
    dict: Dictionary containing confidence matrix and threshold values.
    """
    
    # Extract data from the similarity_dict
    similarity_matrix = similarity_dict['similarity']
    sim_threshold = similarity_dict['threshold']
    
    # Calculate the count of neighbors (users with similarity above threshold) for each user
    count_of_neighbors_for_user = np.sum(similarity_matrix >= sim_threshold, axis=1)
    # Calculate the count of rated items for each user (non-zero entries in the user-item matrix)
    count_of_rated_items_for_user = np.sum(user_item_matrix != 0, axis=1)
    
    # Replace zeros with ones to avoid division by zero
    count_of_neighbors_for_user[count_of_neighbors_for_user == 0] = 1
    count_of_rated_items_for_user[count_of_rated_items_for_user == 0] = 1

    NumberUser = len(user_ids) 

    # Calculate the count of common similar neighbors between users
    common_similar_users_count_matrix = np.sum((similarity_matrix[:, None, :] >= sim_threshold) &
                                               (similarity_matrix[None, :, :] >= sim_threshold), axis=2)

    # Calculate the count of common rated items between users
    common_rated_items_count_matrix = np.sum((user_item_matrix[:, None, :] != 0) &
                                              (user_item_matrix[None, :, :] != 0), axis=2)

    # Calculate the ratio of common similar neighbors to total neighbors for each user pair
    common_similar_neighbors_ratio_matrix = common_similar_users_count_matrix / count_of_neighbors_for_user[:, None]
    # Calculate the ratio of common rated items to total rated items for each user pair
    common_rated_items_ratio_matrix = common_rated_items_count_matrix / count_of_rated_items_for_user[:, None]

    # Initialize a matrix to store confidence values
    confidence_matrix = np.zeros((NumberUser, NumberUser))

    # Calculate the confidence values between user pairs
    for i in range(NumberUser):
        for j in range(NumberUser):
            if i != j:
                common_similar_neighbors_ratio = common_similar_neighbors_ratio_matrix[i, j]
                common_rated_items_ratio = common_rated_items_ratio_matrix[i, j]
                confidence_matrix[i, j] = (common_similar_neighbors_ratio + common_rated_items_ratio) / 2
    
    # Calculate the threshold for confidence values
    positive_count_of_confidence_row = np.count_nonzero(confidence_matrix, axis=0)
    conf_threshold = np.sum(confidence_matrix, axis=0) / positive_count_of_confidence_row
    
    # Create a dictionary containing the calculated confidence matrix and threshold
    confidence_dict = {'confidence': confidence_matrix, 'threshold': conf_threshold}
    
    return confidence_dict


# Main function to calculate identical opinion matrix
def calculate_identical_opinion(user_item_matrix, similarity_dict, confidence_dict):
    """
    Calculates the identical opinion matrix based on similarity, confidence, and rating ratios.

    Parameters:
    user_item_matrix (ndarray): User-item matrix containing ratings.
    similarity_dict (dict): Dictionary containing similarity matrix and threshold values.
    confidence_dict (dict): Dictionary containing confidence matrix and threshold values.

    Returns:
    dict: Dictionary containing identical opinion matrix and threshold values.
    """
    # Numba-compiled function to calculate the ratio of sameness between two data vectors
    @nb.njit
    def calculate_sameness_ratio(data_i, data_j, threshold_i, threshold_j, epsilon):
        # Calculate the sameness ratio between two data vectors based on thresholds and epsilon
        mask_i = data_i > threshold_i 
        mask_j = data_j > threshold_j

        intersected_items = np.intersect1d(np.where(mask_i)[0], np.where(mask_j)[0])
        common_simi_count = len(intersected_items)

        if common_simi_count > 0:
            same_simi_value_count = np.count_nonzero((data_i[intersected_items] - data_j[intersected_items]) <= epsilon)
            sameness_ratio = same_simi_value_count / common_simi_count
        else:
            sameness_ratio = 0

        return sameness_ratio
    
    # Extract data from provided similarity_dict and confidence_dict
    similarity_matrix, sim_threshold = similarity_dict['similarity'], similarity_dict['threshold']
    confidence_matrix, conf_threshold = confidence_dict['confidence'], confidence_dict['threshold']
    epsilon = 0.1  # Threshold for similarity
    
    NumberUser = len(user_ids)  # Number of users
    identical_opinion = np.zeros((NumberUser, NumberUser))  # Initialize identical opinion matrix
    
    # Loop through all pairs of users
    for i in range(NumberUser):
        for j in range(NumberUser):
            if i != j:  # Avoid self-comparison
                # Calculate sameness ratios for similarity, confidence, and item ratings
                similarity_sameness_ratio = calculate_sameness_ratio(similarity_matrix[i], similarity_matrix[j], sim_threshold[i], sim_threshold[j], epsilon)
                confidence_sameness_ratio = calculate_sameness_ratio(confidence_matrix[i], confidence_matrix[j], conf_threshold[i], conf_threshold[j], epsilon)
                rating_sameness_ratio = calculate_sameness_ratio(user_item_matrix[i], user_item_matrix[j], 0, 0, epsilon)

                # Calculate the average of the three sameness ratios as the identical opinion value
                identical_opinion[i, j] = (similarity_sameness_ratio + confidence_sameness_ratio + rating_sameness_ratio) / 3
    
    # Calculate positive counts of identical opinions for each user
    positive_count_of_identopinion_row = np.count_nonzero(identical_opinion, axis=0)
    # Calculate thresholds for identical opinion values
    identopinion_threshold = np.sum(identical_opinion, axis=0) / positive_count_of_identopinion_row
    
    # Create a dictionary to store identical opinion values and thresholds
    identical_opinion_dict = {'opinion': identical_opinion, 'threshold': identopinion_threshold}
    
    return identical_opinion_dict


# Main function to generate incipient trust network
def generate_incipient_trust(similarity_dict, confidence_dict, identical_opinion_dict):
    """
    Generates the incipient trust network based on similarity, confidence, and identical opinion values.

    Parameters:
    similarity_dict (dict): Dictionary containing similarity matrix and threshold values.
    confidence_dict (dict): Dictionary containing confidence matrix and threshold values.
    identical_opinion_dict (dict): Dictionary containing identical opinion matrix and threshold values.

    Returns:
    dict: Dictionary containing incipient trust values, thresholds, and network.
    """
    
    # Calculate direct trust and its threshold
    def calculate_direct_trust(similarity_dict, confidence_dict, identical_opinion_dict):
        # Extract data from provided dictionaries
        similarity, sim_threshold = similarity_dict['similarity'], similarity_dict['threshold']
        confidence, conf_threshold = confidence_dict['confidence'], confidence_dict['threshold']
        opinion, opin_threshold = identical_opinion_dict['opinion'], identical_opinion_dict['threshold']

        # Initialize direct trust matrix and get its dimensions
        direct_trust = np.zeros_like(similarity)
        num_rows, num_cols = similarity.shape

        # Loop through all pairs of users
        for i in range(num_rows):
            for j in range(num_cols):
                # Extract values and thresholds for similarity, confidence, and opinion
                sim_val = similarity[i, j]
                conf_val = confidence[i, j]
                opinion_val = opinion[i, j]
                s_val = sim_threshold[i]
                c_val = conf_threshold[i]
                o_val = opin_threshold[i]

                # Calculate direct trust based on conditions
                if sim_val >= s_val:
                    if conf_val >= c_val:
                        if opinion_val >= o_val:
                            denominator = sim_val + conf_val + opinion_val
                            direct_trust[i, j] = (3 * (sim_val * conf_val * opinion_val)) / denominator
                        else:
                            denominator = sim_val + conf_val
                            direct_trust[i, j] = (2 * (sim_val * conf_val)) / denominator
                    else:
                        if opinion_val >= o_val:
                            denominator = sim_val + opinion_val
                            direct_trust[i, j] = (2 * (sim_val * opinion_val)) / denominator
                else:
                    if conf_val >= c_val:
                        if opinion_val >= o_val:
                            denominator = conf_val + opinion_val
                            direct_trust[i, j] = (2 * (conf_val * opinion_val)) / denominator

        # Calculate positive counts of direct trust for each user
        positive_counts_per_row = np.sum(direct_trust > 0, axis=0)
        dt_threshold = np.sum(direct_trust, axis=0) / np.maximum(positive_counts_per_row, 1) 
        # dt_threshold is the direct trust threshold

        return direct_trust, dt_threshold 

    # Numba-compiled function to propagate direct trust
    @nb.njit
    def propagate_direct_trust(direct_trust, dt_threshold):
        # Initialize propagated trust matrix
        num_users = direct_trust.shape[0]
        propagated_trust = np.zeros((num_users, num_users))

        # Loop through all pairs of users
        for i in range(num_users):
            for j in range(num_users):
                if i != j:
                    dt_i_u = direct_trust[i, :]
                    dt_u_j = direct_trust[:, j]
                    threshold_i = dt_threshold[i]
                    threshold_j = dt_threshold[j]

                    co_trust_sum = 0
                    co_trust_count = 0

                    # Calculate co-trust sum and count for each user u
                    for u in range(num_users):
                        if i != u and j != u and dt_i_u[u] >= threshold_i and dt_u_j[u] >= threshold_j:
                            dt_product = dt_i_u[u] * dt_u_j[u]
                            dt_sum = dt_i_u[u] + dt_u_j[u]
                            co_trust_sum += 2 * dt_product / dt_sum
                            co_trust_count += 1

                    co_trust_count = max(co_trust_count, 1)
                    propagated_trust[i, j] = co_trust_sum / co_trust_count

        return propagated_trust

    # Calculate incipient trust
    def calculate_incipient_trust(direct_trust, propagated_trust):
        non_zero_direct = direct_trust > 0
        non_zero_propagated = propagated_trust > 0

        # Calculate numerator for incipient trust
        numerator = np.where(
            non_zero_direct & non_zero_propagated,
            direct_trust * propagated_trust,
            0
        )

        denominator = direct_trust + propagated_trust
        nonzero_denominator = denominator > 0

        # Calculate incipient trust
        intrust = np.where(
            nonzero_denominator,
            np.divide(numerator, denominator, out=np.zeros_like(denominator), where=nonzero_denominator),
            np.where(
                non_zero_direct,
                propagated_trust,
                np.where(
                    non_zero_propagated,
                    direct_trust,
                    0
                )
            )
        )

        return np.sqrt(intrust)

    NumberUser = len(user_ids)  # Number of users

    # Calculate direct trust with threshold
    direct_trust, dt_threshold = calculate_direct_trust(similarity_dict, confidence_dict, identical_opinion_dict)
    
    # Propagate direct trust between users
    propagated_trust = propagate_direct_trust(direct_trust, dt_threshold)
    
    # Calculate InTrust
    incipient_trust = calculate_incipient_trust(direct_trust, propagated_trust)
    
    # Calculate trust threshold
    positive_counts_per_row = np.sum(incipient_trust > 0, axis=0)
    inctrust_threshold = np.sum(incipient_trust, axis=0) / np.maximum(positive_counts_per_row, 1)
    
    # Create Initial Trust Network
    inctrust_net = incipient_trust >= inctrust_threshold
    
    # Create dictionary containing incipient trust values, thresholds, and network
    incipient_trust_dict = {'trust': incipient_trust, 'threshold': inctrust_threshold, 'network': inctrust_net}
    
    return incipient_trust_dict


# Main function to reconstruct trust to telic form
def reconstruct_incipiet_trust_to_telic(user_item_matrix, incipient_trust_dict):
    """
    Reconstructs the incipient trust values to telic trust values.

    Parameters:
    user_item_matrix (ndarray): User-item matrix containing ratings.
    incipient_trust_dict (dict): Dictionary containing incipient trust values, thresholds, and network.

    Returns:
    dict: Dictionary containing telic trust values, thresholds, and network.
    """
    
    # Calculate trustees' precision
    def calculate_trustees_precision(incipient_trust_dict, user_item_matrix):
        NumberUser, NumberItem = user_item_matrix.shape
        incipient_trust, inctrust_threshold = incipient_trust_dict['trust'], incipient_trust_dict['threshold']
        precision_val = np.zeros((NumberUser, NumberUser))
        
        for i in range(NumberUser):
            # Calculate average rating for user i
            non_zero_ratings_i = user_item_matrix[i, user_item_matrix[i] != 0]
            avg_rating_i = np.mean(non_zero_ratings_i) if non_zero_ratings_i.size > 0 else 0

            for j in range(NumberUser):
                if i != j and incipient_trust[i, j] >= inctrust_threshold[i]:
                    # Calculate average rating for user j
                    non_zero_ratings_j = user_item_matrix[j, user_item_matrix[j] != 0]
                    avg_rating_j = np.mean(non_zero_ratings_j) if non_zero_ratings_j.size > 0 else 0

                    # Find common items rated by both users i and j
                    common_items = np.where((user_item_matrix[i] != 0) & (user_item_matrix[j] != 0))[0]

                    if common_items.size > 0:
                        # Calculate the impact of trustee j's prediction on user i's ratings
                        jth_trustee_prediction_impact = incipient_trust[i, j] * (user_item_matrix[j, common_items] - avg_rating_j)
                        common_items_rating_prediction = avg_rating_i + jth_trustee_prediction_impact

                        errors = np.abs(common_items_rating_prediction - user_item_matrix[i, common_items])
                        fractions = np.maximum(np.abs(MaxRating - user_item_matrix[i, common_items]),
                                               np.abs(MinRating - user_item_matrix[i, common_items]))
                        precision_val[i, j] = np.sum(errors / fractions) / common_items.size
        
        positive_counts_per_row = np.sum(precision_val > 0, axis=0)
        precision_threshold = np.sum(precision_val, axis=0) / np.maximum(positive_counts_per_row, 1)

        return precision_val, precision_threshold
    
   
    # Calculate trustees' precision values and threshold
    precision_val, precision_threshold = calculate_trustees_precision(incipient_trust_dict, user_item_matrix)

    # Create ultimate trust network based on trustees' precision
    telic_trust_net = precision_val >= precision_threshold
    
    # Extract incipient trust values
    incipient_trust = incipient_trust_dict['trust']

    # Calculate telic trust values
    with np.errstate(divide='ignore', invalid='ignore'):
        telic_trust = (incipient_trust * precision_val * 2) / (incipient_trust + precision_val)
        telic_trust[~np.isfinite(telic_trust)] = 0  # Replace inf and NaN with 0

    # Calculate threshold for telic trust
    positive_counts_per_row = np.sum(telic_trust > 0, axis=0)
    telictrust_threshold = np.sum(telic_trust, axis=0) / positive_counts_per_row
    
    # Create dictionary containing telic trust values, thresholds, and network
    telic_trust_dict = {'trust': telic_trust, 'threshold': telictrust_threshold, 'network': telic_trust_net}
    
    return telic_trust_dict


# Main function to prepare data for evaluation
def prepare_evaluation_data(user_ids, user_item_df, telic_trust_dict):
    """
    Prepares data for evaluating trust network performance in rating prediction.

    Parameters:
    user_ids (ndarray): Array of user IDs.
    user_item_df (DataFrame): User-item matrix containing ratings.
    telic_trust_dict (dict): Dictionary containing telic trust values, thresholds, and network.

    Returns:
    ndarray, ndarray, ndarray: Train data, test data, and test trustees' trust values.
    """
    
    # Split data into training and test sets using KFold cross-validation
    def test_train_split(user_item_df, k_folds):
        # Create a KFold object with random_state=42
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        # Split data into train and test sets using KFold cross-validation
        for train_index, test_index in kf.split(user_item_df):
            train_indices, test_indices = train_index, test_index
            train_data, test_data = user_item_df.iloc[train_indices], user_item_df.iloc[test_indices]
        return train_data, test_data
    
    # Split data into train and test sets
    train_data, test_data = test_train_split(user_item_df, k_folds=5)    
    
    # Create a DataFrame for trust values
    trust_df = pd.DataFrame(index=user_ids, columns=user_ids, data=telic_trust_dict['trust'])
    
    # Extract trust values for test trustees
    test_trustees = trust_df.loc[test_data.index][train_data.index]
    test_trustees = np.array(test_trustees)
    
    # Convert train and test data to numpy arrays
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    
    return train_data, test_data, test_trustees


# Main function to evaluate trust network for rating prediction
def evaluate_trust_network(train_data, test_data, test_trustees, metrics_thr):
    """
    Evaluates the trust network performance in rating prediction.

    Parameters:
    train_data (ndarray): Training data for users and items.
    test_data (ndarray): Test data for users and items.
    test_trustees (ndarray): Test trustees' trust values.
    metrics_thr (int): Threshold for classifying predictions.

    Returns:
    float, float, float, float: Precision, recall, MAE, RMSE.
    """
    
    # Identify metrics based on predicted and target ratings
    def identify_metrics(predicted_rating_i, Target_Rating, metrics_thr):
        if Target_Rating >= metrics_thr and predicted_rating_i >= metrics_thr:
            return "TP"
        elif Target_Rating < metrics_thr and predicted_rating_i >= metrics_thr:
            return "FP"
        elif Target_Rating >= metrics_thr and predicted_rating_i < metrics_thr:
            return "FN"
        elif Target_Rating < metrics_thr and predicted_rating_i < metrics_thr:
            return "TN"
        else:
            return None
    
    # Evaluate prediction precision and recall
    def evaluate_prediction_metrics(pred_precision, pred_recall, TP, FP, FN):
        pred_precision = (TP / (TP + FP)) * 100 if TP + FP != 0 else 0
        pred_recall = (TP / (TP + FN)) * 100 if TP + FN != 0 else 0
        return pred_precision, pred_recall
    
    num_test = len(test_data)
    TP = FP = FN = TN = MAE_diff = RMSE_diff = 0
    
    # Loop through each test user
    for u in range(num_test):
        trustees_ids = np.nonzero(test_trustees[u])[0]
        trustees_weights = test_trustees[u, trustees_ids]

        rated_movies_ids = np.nonzero(test_data[u])[0]
        rated_movies_ratings = test_data[u, rated_movies_ids]
        rating_avg_u = rated_movies_ratings.mean()

        neighbors = train_data[trustees_ids, :]
        num_trustees = len(trustees_ids)

        # Loop through each rated movie
        for i in rated_movies_ids:
            prediction_numerator = 0

            # Loop through trustees to calculate prediction numerator
            for v in range(num_trustees):
                trustee_rating = neighbors[v, i]

                if trustee_rating > 0:
                    trustee_avg = neighbors[v, np.nonzero(neighbors[v])].mean()
                    prediction_numerator += trustees_weights[v] * (trustee_rating - trustee_avg)

            weights_sum = trustees_weights.sum()
            if weights_sum != 0:
                predicted_rating_i = rating_avg_u + (prediction_numerator / weights_sum)
                
                # Validate predicted rating and calculate evaluation metrics
                Target_Rating = test_data[u, i]
                if Target_Rating != 0 and predicted_rating_i > 0:
                    metric = identify_metrics(predicted_rating_i, Target_Rating, metrics_thr)
                    if metric == "TP":
                        TP += 1
                    elif metric == "FP":
                        FP += 1
                    elif metric == "FN":
                        FN += 1
                    elif metric == "TN":
                        TN += 1

                    MAE_diff += abs(predicted_rating_i - Target_Rating)
                    RMSE_diff += (predicted_rating_i - Target_Rating) ** 2
    
    # Calculate prediction precision, recall, MAE, RMSE, and elapsed time
    pred_precision, pred_recall = evaluate_prediction_metrics(0, 0, TP, FP, FN)
    eval_num = TP + FP + FN + TN
    pred_MAE = MAE_diff / eval_num if eval_num != 0 else 0
    pred_RMSE = np.sqrt(RMSE_diff / eval_num) if eval_num != 0 else 0
 
    return pred_precision, pred_recall, pred_MAE, pred_RMSE


# Main function
def main():
    """
    The main function to execute the entire process of ITN-based collaborative filtering for rating prediction.
    """
    
    # Load the data
    data_file_path = 'data/Data.csv'
    df = pd.read_csv(data_file_path)

    # Extract unique movie and user IDs
    movie_ids = np.sort(df['movieId'].unique())
    
    global user_ids
    user_ids = np.sort(df['userId'].unique())
    #global user_ids
    
    global MaxRating, MinRating
    MaxRating, MinRating = 5, 1
    #global MaxRating, MinRating
    
    start_time = time.time()
    
    user_item_df = create_user_item_df(df)
    user_item_matrix = user_item_df.to_numpy()
    # calculate proposed similarity, condidence, and identical opinion matrixes by coresponding function
    similarity_dict = calculate_similarity(user_item_matrix)
    confidence_dict = calculate_confidence(user_item_matrix, similarity_dict)
    identical_opinion_dict = calculate_identical_opinion(user_item_matrix, similarity_dict, confidence_dict)
    
    # Calculate and construct proposed trust
    incipient_trust_dict = generate_incipient_trust(similarity_dict, confidence_dict, identical_opinion_dict)
    telic_trust_dict = reconstruct_incipiet_trust_to_telic(user_item_matrix, incipient_trust_dict) #user_ids, MinRating=1, MaxRating=5)

    # preparing data for evaluating the proposed trust performance in rating prediction
    '''this include 
         - creating train and test data using K-Fold method (here it is fixed at K=5)
         - prepare the trust matrix of test set
         - converting the data into numpy array
    '''
    train_data, test_data, test_trustees = prepare_evaluation_data(user_ids, user_item_df, telic_trust_dict)
    
    # evaluating proposed trust method (ITN) for rating prediction 
    metrics_threshold = 4 # this is used for classifying predictions results by confusion matrix
    '''It is set at 4 for Movielens dataset that the rating scale is in range 1 to 5 
       (1 and 2 indicate lower interested, 3 indecates so so (maybe interested), and 4 and 5 indecate high interested (5 highest level of interested).
       You should set it respecting the rating scale of dataset you use
    '''
    pred_precision, pred_recall, pred_MAE, pred_RMSE = evaluate_trust_network(train_data, test_data, test_trustees, metrics_threshold)

    run_time = time.time() - start_time
    
    print('Run Time: ', run_time)
    print(' ')
    print('Results: ')
    print(' ')
    print('Precision: ', pred_precision)
    print(' ')
    print('Recall: ', pred_recall)
    print(' ')
    print('MAE: ', pred_MAE)
    print(' ')
    print('RMSE: ', pred_RMSE)
    

if __name__ == '__main__':
    main()