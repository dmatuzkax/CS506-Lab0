import pytest
from utils import *

def test_dot_product():
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    
    result = dot_product(vector1, vector2)
    assert result == 32, f"Expected 32, but got {result}"
    
def test_cosine_similarity():
    ### YOUR CODE HERE
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    result = cosine_similarity(vector1, vector2)
    expected_result = 32 / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    assert np.isclose(result, expected_result), f"Expected {expected_result}, but got {result}"

def test_nearest_neighbor():
    ### YOUR CODE HERE
    points = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8]
    ])
    query_point = np.array([3, 4])
    result = nearest_neighbor(points, query_point)
    expected_index = 7
    assert result == expected_index, f"Expected index {expected_index}, but got {result}"
