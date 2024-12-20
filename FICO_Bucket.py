import numpy as np
import pandas as pd

# Helper function to compute Mean Squared Error (MSE) for a given bucket
def compute_mse(scores):
    mean_value = np.mean(scores)
    mse = np.mean((scores - mean_value) ** 2)
    return mse

# Helper function to compute Log-Likelihood (LL) for a given bucket
def compute_log_likelihood(defaults, non_defaults):
    k = np.sum(defaults)
    n = k + np.sum(non_defaults)
    if k == 0 or n == 0 or k == n:
        return 0  # To avoid log(0) or division by zero errors
    p = k / n
    ll = k * np.log(p) + (n - k) * np.log(1 - p)
    return ll

# Dynamic programming function for finding optimal bucket boundaries
def optimal_buckets(fico_scores, defaults, num_buckets, objective='mse'):
    """
    Parameters:
    - fico_scores: List or array of FICO scores.
    - defaults: Binary list indicating if borrower defaulted (1) or not (0).
    - num_buckets: Number of buckets to create.
    - objective: Either 'mse' (minimize Mean Squared Error) or 'll' (maximize Log-Likelihood).
    
    Returns:
    - bucket boundaries that optimize the given objective.
    """
    n = len(fico_scores)
    
    # Sort FICO scores and corresponding defaults together
    sorted_indices = np.argsort(fico_scores)
    fico_scores = np.array(fico_scores)[sorted_indices]
    defaults = np.array(defaults)[sorted_indices]
    
    # Initialize DP table: dp[i][j] will store the minimum cost for dividing the first i scores into j buckets
    dp = np.full((n + 1, num_buckets + 1), np.inf)
    dp[0][0] = 0
    
    # Store the boundaries of the buckets
    boundaries = np.zeros((n + 1, num_buckets + 1), dtype=int)
    
    # Precompute sums and counts for efficiency
    sums = np.cumsum(fico_scores)
    sums_sq = np.cumsum(fico_scores ** 2)
    defaults_sum = np.cumsum(defaults)
    non_defaults_sum = np.cumsum(1 - defaults)
    
    # Helper function to compute the cost of a bucket from [start, end]
    def bucket_cost(start, end):
        if objective == 'mse':
            # Use MSE: Mean Squared Error
            bucket_scores = fico_scores[start:end + 1]
            return compute_mse(bucket_scores)
        elif objective == 'll':
            # Use Log-Likelihood
            bucket_defaults = defaults_sum[end] - (defaults_sum[start - 1] if start > 0 else 0)
            bucket_non_defaults = (non_defaults_sum[end] - (non_defaults_sum[start - 1] if start > 0 else 0))
            return -compute_log_likelihood(bucket_defaults, bucket_non_defaults)  # Minimizing -LL
    
    # Fill DP table using dynamic programming
    for j in range(1, num_buckets + 1):  # Number of buckets
        for i in range(1, n + 1):  # Number of scores considered
            for k in range(i):  # Previous bucket boundary
                cost = dp[k][j - 1] + bucket_cost(k, i - 1)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    boundaries[i][j] = k
    
    # Retrieve the bucket boundaries
    bucket_boundaries = []
    idx = n
    for j in range(num_buckets, 0, -1):
        bucket_boundaries.append(boundaries[idx][j])
        idx = boundaries[idx][j]
    
    bucket_boundaries.reverse()
    
    return bucket_boundaries, dp[n][num_buckets]

# Function to bucket FICO scores and return the categorized data
def bucket_fico_scores(fico_scores, defaults, num_buckets, objective='mse'):
    """
    Buckets FICO scores into the given number of buckets and assigns categorical labels.
    
    Parameters:
    - fico_scores: List or array of FICO scores.
    - defaults: List or array of defaults (binary values, 1 = default, 0 = no default).
    - num_buckets: Number of buckets to create.
    - objective: Either 'mse' or 'll' to optimize Mean Squared Error or Log-Likelihood, respectively.
    
    Returns:
    - bucketed_scores: Array of bucketed FICO scores.
    - bucket_boundaries: The boundaries that define each bucket.
    """
    # Get the optimal bucket boundaries using dynamic programming
    bucket_boundaries, _ = optimal_buckets(fico_scores, defaults, num_buckets, objective)
    
    # Bucket the FICO scores based on the boundaries
    bucketed_scores = np.digitize(fico_scores, bucket_boundaries)
    
    return bucketed_scores, bucket_boundaries

# Function to load data from a CSV and apply bucketing
def load_and_bucket_data(csv_file, fico_col, default_col, num_buckets, objective='mse'):
    """
    Loads the data from a CSV file, processes it, and applies bucketing to the FICO scores.
    
    Parameters:
    - csv_file: Path to the CSV file.
    - fico_col: Name of the column containing FICO scores.
    - default_col: Name of the column indicating default (1 = default, 0 = no default).
    - num_buckets: Number of buckets to create.
    - objective: Either 'mse' (minimize Mean Squared Error) or 'll' (maximize Log-Likelihood).
    
    Returns:
    - DataFrame with original data and the bucketed FICO scores.
    - List of bucket boundaries.
    """
    # Load the data
    data = pd.read_csv(csv_file)
    
    # Extract FICO scores and defaults
    fico_scores = data[fico_col].values
    defaults = data[default_col].values
    
    # Bucket the FICO scores
    bucketed_scores, bucket_boundaries = bucket_fico_scores(fico_scores, defaults, num_buckets, objective)
    
    # Add bucketed scores to the DataFrame
    data['bucketed_fico'] = bucketed_scores
    
    return data, bucket_boundaries

# Example Usage:
if __name__ == "__main__":
    # Path to the CSV file
    csv_file = 'C:/Forage/JP - Quantitative Research/Task 3 and 4_Loan_Data.csv'  # Replace with your actual file path

    # Define the columns in the CSV file
    fico_col = 'fico_score'  # Column name for FICO scores
    default_col = 'default'  # Column name for default status (1 = default, 0 = no default)
    
    # Define the number of buckets
    num_buckets = 5  # Example: divide into 5 buckets
    
    # Define the optimization objective ('mse' or 'll')
    objective = 'mse'  # 'mse' for Mean Squared Error, 'll' for Log-Likelihood
    
    # Load data and bucket FICO scores
    bucketed_data, bucket_boundaries = load_and_bucket_data(csv_file, fico_col, default_col, num_buckets, objective)
    
    # Print results
    print(bucketed_data.head())
    print("Bucket Boundaries:", bucket_boundaries)
