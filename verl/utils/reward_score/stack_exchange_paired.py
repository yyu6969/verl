def compute_score(response, chosen, rejected):
    """The scoring function for stack-exchange-paired dataset.
    
    This function computes a reward score based on how similar the generated response
    is to the chosen response compared to the rejected response.
    
    Args:
        response: the generated response to score
        chosen: the preferred/chosen response from the dataset
        rejected: the rejected/non-preferred response from the dataset
        
    Returns:
        float: A reward score where:
        - Higher positive values indicate the response is more similar to the chosen response
        - Lower negative values indicate the response is more similar to the rejected response
        - 0 indicates neutral/equal similarity to both
    """
    # For initial implementation, we can use a simple binary reward:
    # 1.0 if the response exactly matches the chosen response
    # -1.0 if the response exactly matches the rejected response
    # 0.0 otherwise
    
    if response == chosen:
        return 1.0
    elif response == rejected:
        return -1.0
    return 0.0

# Note: This is a basic implementation. For better results, consider:
# 1. Using semantic similarity metrics (e.g., embedding distance)
# 2. Implementing partial matching rewards
# 3. Adding additional reward components for response quality
