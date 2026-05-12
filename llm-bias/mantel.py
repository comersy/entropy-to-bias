import numpy as np
from scipy.stats import pearsonr, spearmanr

def matrix_to_vector(matrix: np.ndarray) -> np.ndarray:
    """
    Extract upper triangle of a distance matrix as a flat vector.
    """
    n = matrix.shape[0]
    indices = np.triu_indices(n, k=1)
    return matrix[indices]


def mantel_test(matrix1: np.ndarray, matrix2: np.ndarray, 
                n_permutations: int = 999, 
                method: str = 'pearson') -> tuple:
    """
    Mantel test: correlation between two distance matrices.
    """
    corr_fn = pearsonr if method == 'pearson' else spearmanr

    vec1 = matrix_to_vector(matrix1)
    vec2 = matrix_to_vector(matrix2)
    observed_r, _ = corr_fn(vec1, vec2)

    n = matrix1.shape[0]
    permuted_rs = []

    for _ in range(n_permutations):
        perm = np.random.permutation(n)
        matrix1_permuted = matrix1[np.ix_(perm, perm)]
        vec1_permuted = matrix_to_vector(matrix1_permuted)
        r_perm, _ = corr_fn(vec1_permuted, vec2)
        permuted_rs.append(r_perm)

    permuted_rs = np.array(permuted_rs)
    p_value = np.mean(permuted_rs >= observed_r)

    return observed_r, p_value, permuted_rs


def mantel_summary(name1: str, name2: str, r: float, p_value: float) -> None:
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"{name1} vs {name2}")
    print(f"  r = {r:.4f}  |  p = {p_value:.4f}  {significance}")