import numpy as np
import math
import matplotlib.pyplot as plt

# --- 1. Hàm simulate_price ---
def simulate_price(f0: float, mu: float, sigma: float, days: int = 390, seed: int = None) -> np.ndarray:
    """
    Mô phỏng chuỗi giá tài sản lý tưởng (f) bằng mô hình Geometric Brownian Motion (GBM)
    sử dụng phương pháp Euler.

    Args:
        f0 (float): Giá tài sản ban đầu.
        mu (float): Tỷ suất sinh lời kỳ vọng hàng ngày.
        sigma (float): Độ biến động hàng ngày.
        days (int): Số ngày mô phỏng.
        seed (int, optional): Giá trị seed cho bộ tạo số ngẫu nhiên (để tái tạo kết quả).
    Returns:
        np.ndarray: Mảng chứa chuỗi giá tài sản lý tưởng có độ dài 'days'.
    """
    if seed is not None:
        np.random.seed(seed)

    eps = np.random.normal(loc=0.0, scale=1.0, size=days)
    prices = np.empty(days + 1, dtype=float)
    prices[0] = f0

    for t in range(days):
        prices[t+1] = prices[t] * (1 + mu + sigma * eps[t])

    return prices[1:] # Trả về N phần tử (từ ngày 1 đến ngày N)

# --- 2. Hàm pandemic_impact ---
def pandemic_impact(N: int, p_percent: float, L: int, I: int) -> np.ndarray:
    """
    Tính toán yếu tố tác động của đại dịch (p) theo thời gian.
    Mô hình hóa bằng sự kết hợp của hàm sin và cosin.

    Args:
        N (int): Tổng số bước thời gian (ngày).
        p_percent (float): Phần trăm tác động tối đa của đại dịch (ví dụ: 1 cho 1%).
        L (int): Chiều dài của giai đoạn tác động ban đầu (sử dụng hàm sin).
        I (int): Chiều dài của giai đoạn phục hồi (sử dụng hàm cosin).
    Returns:
        np.ndarray: Mảng chứa các giá trị p theo thời gian có độ dài 'N'.
    """
    p_values = []
    for t in range(N):
        if t < L:
            p = -(p_percent / 100) * math.sin(math.pi * t / L)
        elif t < L + I:
            p = -(p_percent / 100) * math.cos(math.pi * (t - L) / I)
        else:
            p = 0
        p_values.append(p)
    return np.array(p_values)

# --- 3. Hàm calculate_P ---
def calculate_P(N: int, f0: float, mu: float, sigma: float, alpha: float, 
                p_percent: float, L: int, I: int, epsilon: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tính toán giá tài sản thực tế (P) dưới tác động của đại dịch.
    P[t] = f[t] + alpha * p[t], tính bằng vòng lặp tường minh.

    Args:
        N (int): Tổng số ngày mô phỏng.
        f0 (float): Giá tài sản ban đầu.
        mu (float): Tỷ suất sinh lời kỳ vọng hàng năm (sẽ được chia cho N để thành daily).
        sigma (float): Độ biến động hàng ngày.
        alpha (float): Hệ số nhạy cảm của giá tài sản với tác động đại dịch.
        p_percent (float): Tỉ lệ xuất hiện đại dịch.
        L (int): Giai đoạn tác động ban đầu.
        I (int): Cường độ đại dịch (giai đoạn phục hồi).
        epsilon (float, optional): Một giá trị nhỏ để tránh giá về 0. Mặc định là 1e-6.
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple chứa (P_values, f_values, p_values).
    """
    # Bước 1: Tính toán toàn bộ mảng _price (f)
    # daily_mu được chia cho N để chuyển đổi từ annual sang daily nếu mu là annual
    _price = simulate_price(f0=f0, mu=mu/N, sigma=sigma, days=N)

    # Bước 2: Tính toán toàn bộ mảng _impact (p)
    _impact = pandemic_impact(N=N, p_percent=p_percent, L=L, I=I)

    print(f"Len of _price: {len(_price)}")
    print(f"Len of _impact: {len(_impact)}")

    # Bước 3: Tính toán mảng P bằng vòng lặp tường minh
    P_values = []
    if len(_price) != N or len(_impact) != N:
        raise ValueError(f"Kích thước mảng f ({len(_price)}) hoặc p ({len(_impact)}) không khớp với N ({N}).")

    for t in range(N):
        P_t = _price[t] + alpha * _impact[t] # P[t] = f[t] + alpha * p[t]
        P_values.append(P_t)

    P = np.array(P_values) # Chuyển danh sách P_values thành mảng NumPy

    return P, _price, _impact

# --- 4. Hàm plot_P_and_f ---
def plot_P_and_f(P_values: np.ndarray, f_values: np.ndarray, title: str = "Biểu đồ Giá tài sản thực tế (P) và Giá lý tưởng (f)"):
    """
    Vẽ biểu đồ của P và f trên cùng một đồ thị.

    Args:
        P_values (np.ndarray): Mảng giá trị P (giá thực tế).
        f_values (np.ndarray): Mảng giá trị f (giá lý tưởng).
        title (str, optional): Tiêu đề của biểu đồ. Mặc định là "Biểu đồ Giá tài sản thực tế (P) và Giá lý tưởng (f)".
    """
    N = len(P_values)
    if len(f_values) != N:
        print("Cảnh báo: Các mảng P và f có độ dài khác nhau. Biểu đồ có thể không chính xác.")
        N = min(N, len(f_values))
        P_values = P_values[:N]
        f_values = f_values[:N]

    days = np.arange(N) # Tạo trục x

    plt.figure(figsize=(14, 7)) # Kích thước biểu đồ lớn hơn để dễ nhìn
    plt.plot(days, P_values, label='P (Giá thực tế)', color='blue', linewidth=2)
    plt.plot(days, f_values, label='f (Giá lý tưởng)', color='green', linestyle='--')

    plt.title(title)
    plt.xlabel('Ngày')
    plt.ylabel('Giá trị')
    plt.legend() # Hiển thị chú giải
    plt.grid(True) # Bật lưới
    plt.tight_layout() # Điều chỉnh bố cục để mọi thứ vừa vặn
    plt.show()

# --- 5. Phần chạy chính của chương trình (Ví dụ sử dụng module) ---
if __name__ == "__main__":
    print("--- Bắt đầu chạy mô phỏng từ module ---")

    # Các tham số mô phỏng
    N_days = 390          # Tổng số ngày mô phỏng
    initial_f = 100.0     # Giá tài sản ban đầu lý tưởng
    daily_mu = 0.02       # Tỷ suất sinh lời kỳ vọng hàng năm (2%)
    daily_sigma = 0.01    # Độ biến động hàng ngày (1%)

    alpha_sensitivity = 0.4 # Hệ số nhạy cảm với đại dịch (alpha)
    p_max_percent = 1     # Tỉ lệ xuất hiện đại dịch tại ngày bất kỳ (1 tương ứng với 1%)
    L_phase = 240         # Giai đoạn tác động ban đầu của đại dịch (ngày)
    I_phase = 50          # Cường độ đại dịch (giai đoạn phục hồi)

    try:
        # Gọi hàm calculate_P từ module để tính toán các chuỗi giá trị
        P_simulation, f_simulation, p_simulation = calculate_P(
            N=N_days,
            f0=initial_f,
            mu=daily_mu,
            sigma=daily_sigma,
            alpha=alpha_sensitivity,
            p_percent=p_max_percent,
            L=L_phase,
            I=I_phase
        )
        print("--- Tính toán hoàn tất thành công ---")
        print(f"Mảng P cuối cùng có {len(P_simulation)} phần tử.")

        # Gọi hàm plot_P_and_f để vẽ biểu đồ
        plot_P_and_f(P_simulation, f_simulation)

        # Nếu bạn muốn, bạn có thể thêm các biểu đồ khác hoặc phân tích dữ liệu tại đây
        # Ví dụ: vẽ riêng tác động p
        # plt.figure(figsize=(14, 7))
        # plt.plot(np.arange(N_days), p_simulation * 100, label='p (Tác động đại dịch x100)', color='red')
        # plt.title('Xu hướng tác động đại dịch (p)')
        # plt.xlabel('Ngày')
        # plt.ylabel('Giá trị')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình mô phỏng: {e}")

    print("--- Kết thúc script mô phỏng ---")
    