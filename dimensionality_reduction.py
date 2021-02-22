import numpy as np
import pandas as pd

def svd(arr, ratio=100):
    u, s, vt = np.linalg.svd(arr)
    print(u.shape, s.shape, vt.shape)

    idx = np.where(s >= np.percentile(s, 100 - ratio))[0][-1] + 1 # 찾고 싶은 백분위 인덱스: 만약 100이면 모든 정보 보존
    reconstructed_matrix = pd.DataFrame(u[:, :idx] @ np.diag(s[:idx]) @ vt[:idx, :])
    print(f"{ratio}%만큼 특잇값 선택하여 복원: {reconstructed_matrix.shape}")

    return reconstructed_matrix