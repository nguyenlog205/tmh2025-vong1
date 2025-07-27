import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.scenario import calculate_P
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"BASE_DIR: {BASE_DIR}")

def generate_dataset(
        N: int,
        f10: float, mu1: float, sigma1: float, alpha1: float,
        f20: float, mu2: float, sigma2: float, alpha2: float,
        f30: float, mu3: float, sigma3: float, alpha3: float,
        p_percent: float, L: int, I: int, epsilon: float = 1e-6
) -> pd.DataFrame:
    """
    Sinh ra dataset gồm 3 cột tương ứng với 3 loại tài sản có mô hình biến động riêng biệt.
    Mỗi tài sản được mô phỏng dựa trên hàm calculate_P.
    """
    P1, _, _ = calculate_P(N, f10, mu1, sigma1, alpha1, p_percent, L, I, epsilon)
    P2, _, _ = calculate_P(N, f20, mu2, sigma2, alpha2, p_percent, L, I, epsilon)
    P3, _, _ = calculate_P(N, f30, mu3, sigma3, alpha3, p_percent, L, I, epsilon)

    df = pd.DataFrame({
        "Asset_1": P1,
        "Asset_2": P2,
        "Asset_3": P3
    })

    return df

