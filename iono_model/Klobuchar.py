import numpy as np

def calculate_klobuchar(ipp_lat, ipp_lon, t_gps, alpha, beta):
    """
    Klobuchar电离层广播模型 (针对网格点 VTEC 的对比简化版本)
    直接利用网格点/穿刺点(IPP)的经纬度计算垂直电离层延迟补偿时间 (相当于 VTEC)。
    
    参数:
    ----------
    ipp_lat : float
        网格点/穿刺点的大地纬度 (单位: 度 [-90, +90])
    ipp_lon : float
        网格点/穿刺点的大地经度 (单位: 度 [-180, +180])
    t_gps : float
        观测时刻的 GPS 周内秒 (GPS Time of Week， 范围 0~604800)
    alpha : list or numpy.ndarray
        广播星历历书中的 alpha 系数 (振幅相关) [a0, a1, a2, a3]
    beta : list or numpy.ndarray
        广播星历历书中的 beta 系数 (周期相关) [b0, b1, b2, b3]
        
    返回:
    ----------
    I_v_sec : float
        垂直电离层网格点群延迟时间 (Vertical Ionospheric Delay)，单位: 秒
    """
    
    # 1. 穿刺点(网格点)维度单位转换为半圆 (Semicircles) 
    semi_circle = 180.0
    phi_i = ipp_lat / semi_circle
    lambda_i = ipp_lon / semi_circle
    
    # 根据 GPS ICD 规范限制纬度极值范围在 [-0.416, 0.416] 半圆之间
    if phi_i > 0.416:
        phi_i = 0.416
    elif phi_i < -0.416:
        phi_i = -0.416
        
    # 2. 根据穿刺点找地磁纬度 phi_m (以偶极子模型近似)，单位: 半圆
    # 其中 0.064 与 1.617 分别来自地磁倾角近似常数
    phi_m = phi_i + 0.064 * np.cos((lambda_i - 1.617) * np.pi)
    
    # 3. 计算穿刺点的当地时间 (Local Time) t，单位: 秒
    # 43200是12小时秒数，用于将半圆转为时间相角
    t = 43200.0 * lambda_i + t_gps
    # 将时间严格规范到一天之内范围 (0 ~ 86400秒)
    t = t % 86400.0
    if t < 0:
        t += 86400.0
        
    # 4. 计算垂直电离层延迟量的幅度 A (Amplitude)并限制下界
    A = sum([alpha[i] * (phi_m ** i) for i in range(4)])
    if A < 0.0:
        A = 0.0
        
    # 5. 计算垂直电离层延迟量的周期 P (Period)并限制下界
    P = sum([beta[i] * (phi_m ** i) for i in range(4)])
    if P < 72000.0:
        P = 72000.0
        
    # 6. 计算当前时间的余弦相位量 x，以弧度为单位计算
    # 夜间模型恒定补偿(Phase超出1.57弧度即表示进入子夜常数段)
    x = (2.0 * np.pi * (t - 50400.0)) / P
    
    # 7. 去掉接收机视向倾角模型，直接求解基础垂直延迟补偿总量 I_v_sec (单位: 秒)
    # 常量背景电离层影响: 5.0e-9 秒 (约相等于 1.5 米)
    if abs(x) < 1.57:
        I_v_sec = (5.0e-9 + A * (1.0 - (x**2)/2.0 + (x**4)/24.0))
    else:
        # 当相位超过 1.57 即属于只有较低自由电子的常数恒定背景阶段
        I_v_sec = 5.0e-9
        
    return I_v_sec

def klobuchar_to_vtec(I_v_sec, freq=1575.42e6):
    """
    Klobuchar 模型的时间延迟(秒)直接转为 VTEC 垂直电子总量。
    由于单频 L1 C/A 是标准的参考基准，通常以此直接逆换算。
    
    参数:
    ----------
    I_v_sec : float
        提取出的垂直方向空间时间延迟量 (s)
    freq : float
        系统主载波频率 (Hz) -> L1 对应 1575.42 MHz。
        
    返回:
    ----------
    vtec_tecu : float
        对应网格点处的 VTEC 值 (单位 TECU)
    """
    c = 299792458.0  # 光速 m/s
    delay_meters = I_v_sec * c
    
    # 标准单频色散公式: distance_delay = 40.3 * STEC_amount / freq^2
    vtec_amount = (delay_meters * (freq ** 2)) / 40.3
    
    # 1 TECU 等价于 10^16 个 free electrons / m^2
    vtec_tecu = vtec_amount / 1.0e16
    
    return vtec_tecu


if __name__ == "__main__":
    # ====== 用法演示与小测试 (参数以 2004某天历书常见参数为参考) ======
    
    # a0 a1 a2 a3 (量纲和幅度相关)
    alpha_exam = [0.1118e-07, -0.7451e-08, -0.5960e-07, 0.1192e-06]
    # b0 b1 b2 b3 (量纲和周期波动相关)
    beta_exam  = [0.1167e+06, -0.2294e+06, -0.1311e+06, 0.1049e+07]
    
    # 假设深度学习网格输出的一点中心 IPP 经纬度
    ipp_lat = 40.0   # 网格点穿刺纬度 40 (°N)
    ipp_lon = 116.0  # 网格点穿刺经度 116 (°E)
    time_w = 345600  # 某GPS周内秒数(例如周四正午等，供测算用)
    
    # ========= 计算执行 =========
    v_delay_sec = calculate_klobuchar(ipp_lat, ipp_lon, time_w, alpha_exam, beta_exam)
    vtec_val = klobuchar_to_vtec(v_delay_sec)
    
    print(f"|--- 网格 VTEC 对比版本 Klobuchar 估算 ---|")
    print(f"网格点/IPP位置: 纬度 {ipp_lat:.1f}°N, 经度 {ipp_lon:.1f}°E")
    print(f"GPS时限: {time_w} sec (Time of Week)")
    print("-" * 40)
    print(f"垂直电离层时间延迟: {v_delay_sec:.14f} 秒")
    print(f"等效垂直 VTEC: {vtec_val:.4f} TECU")
