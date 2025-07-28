import sys, os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.scenario import calculate_P

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"BASE_DIR: {BASE_DIR}")

def generate_dataset(
        N: int,
        f10: float, mu1: float, sigma1: float, alpha1: float,
        f20: float, mu2: float, sigma2: float, alpha2: float,
        f30: float, mu3: float, sigma3: float, alpha3: float,
        p_percent: float, L: int, I: int,
        start_date: str = "2024-01-01",
        epsilon: float = 1e-6
) -> pd.DataFrame:
    """
    Sinh ra dataset gồm 3 cột tương ứng với 3 loại tài sản có mô hình biến động riêng biệt.
    Thêm cột ngày (Date) bắt đầu từ start_date.

    Returns:
        DataFrame với các cột: Date, Asset_1, Asset_2, Asset_3
    """
    P1, _, _ = calculate_P(N, f10, mu1, sigma1, alpha1, p_percent, L, I, epsilon)
    P2, _, _ = calculate_P(N, f20, mu2, sigma2, alpha2, p_percent, L, I, epsilon)
    P3, _, _ = calculate_P(N, f30, mu3, sigma3, alpha3, p_percent, L, I, epsilon)

    # Sinh cột ngày từ start_date
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    date_list = [start_dt + timedelta(days=i) for i in range(N)]

    df = pd.DataFrame({
        "Date": date_list,
        "Asset_1": P1,
        "Asset_2": P2,
        "Asset_3": P3
    })

    return df
