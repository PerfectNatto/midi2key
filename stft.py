def custom_stft(x, n_fft=256, hop_length=None, window=None):
    if hop_length is None:
        hop_length = n_fft // 4
    if window is None:
        window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
    
    # フレーム分割とウィンドウ適用
    frames = x.unfold(-1, n_fft, hop_length) * window  # [batch, n_fft, n_frames]
    
    # 周波数ビンの作成
    k = torch.arange(n_fft//2 + 1, device=x.device).float().unsqueeze(1)  # [n_freq, 1]
    n = torch.arange(n_fft, device=x.device).float().unsqueeze(0)      # [1, n_fft]
    theta = 2 * torch.pi * k * n / n_fft                             # [n_freq, n_fft]
    
    # DFT行列の実部と虚部を計算
    real_matrix = torch.cos(theta)  # [n_freq, n_fft]
    imag_matrix = -torch.sin(theta) # [n_freq, n_fft]
    
    # 実部と虚部の計算
    real_part = torch.matmul(frames, real_matrix.t())  # [batch, n_freq, n_frames]
    imag_part = torch.matmul(frames, imag_matrix.t())  # [batch, n_freq, n_frames]
    
    # 実部と虚部を最後の次元に結合
    return torch.stack([real_part, imag_part], dim=-1)  # [batch, n_freq, n_frames, 2]
