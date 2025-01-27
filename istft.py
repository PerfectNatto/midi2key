import torch
import math

def ori_istft(z, n_fft, hop_length, window, win_length, normalized=True, length=None, center=True):
    real, imag = z[..., 0], z[..., 1]  # 実部と虚部を分割
    freqs, frames = real.shape[-2], real.shape[-1]
    
    # 時間サンプルと周波数ビンの角度を計算
    t = torch.arange(n_fft, device=z.device).unsqueeze(0)  # [1, n_fft]
    k = torch.arange(freqs, device=z.device).unsqueeze(1)  # [freqs, 1]
    angles = 2 * math.pi * k * t / n_fft
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    
    # 時間領域フレームを計算
    x_frames = real.unsqueeze(-1) * cos_angles - imag.unsqueeze(-1) * sin_angles
    x_frames = x_frames * window  # 窓関数を適用
    
    # 出力信号の長さを設定
    if hop_length is None:
        hop_length = win_length // 4
    output_length = hop_length * (frames - 1) + n_fft
    if length:
        output_length = length
    
    # 出力バッファを初期化
    x = torch.zeros(z.shape[:-2] + (output_length,), device=z.device, dtype=z.dtype)
    
    # オーバーラップ・アド
    for i in range(frames):
        start = i * hop_length
        end = start + n_fft
        if end > output_length:
            x[..., start:output_length] += x_frames[..., i, :output_length - start]
        else:
            x[..., start:end] += x_frames[..., i, :]
    
    return x[..., :length] if length else x
