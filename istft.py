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

import torch
import torch.nn as nn
import torch.nn.functional as F

def istft_with_conv_transpose(x_frames, n_fft, hop_length, window):
    """
    x_frames: [batch, frames, n_fft]  -- 各フレームに対して窓関数が適用済み
    n_fft: FFT サイズ
    hop_length: フレーム間のシフト
    window: [n_fft] のウィンドウ関数
    """
    batch, frames, _ = x_frames.shape
    # 入力を [batch, 1, frames * n_fft] にreshapeするわけではなく、
    # 各フレームを独立したチャネルとして扱います
    x_frames = x_frames.transpose(1, 2)  # [batch, n_fft, frames]
    
    # ConvTranspose1d の入力チャネルは n_fft、出力チャネルは 1、カーネルサイズは n_fft
    # ストライドを hop_length に設定
    conv_transpose = nn.ConvTranspose1d(
        in_channels=x_frames.shape[1],
        out_channels=1,
        kernel_size=n_fft,
        stride=hop_length,
        bias=False
    )
    
    # カーネルの重みを固定：各フィルタはウィンドウ関数を反転させたもの（または適切な重み）に設定
    # ここでは、シンプルな例として、ダイレクトに window を設定（適宜調整が必要）
    with torch.no_grad():
        weight = window.flip(0).unsqueeze(0).unsqueeze(0)  # [1, 1, n_fft]
        # 重みの形状は [out_channels, in_channels, kernel_size] となるので、
        # 全チャネルに同じウィンドウを適用する例
        conv_transpose.weight.copy_(weight.expand(conv_transpose.out_channels, x_frames.shape[1], n_fft))
    
    # ConvTranspose1d の実行
    x_reconstructed = conv_transpose(x_frames)  # [batch, 1, output_length]
    return x_reconstructed.squeeze(1)  # [batch, output_length]

# 使用例（概念例）
batch = 2
frames = 10
n_fft = 2048
hop_length = 1024
window = torch.hann_window(n_fft)

# 例として、ランダムなフレーム群（すでに窓関数適用済みとする）
x_frames = torch.randn(batch, frames, n_fft)

# ISTFT を ConvTranspose1d で実行
reconstructed_signal = istft_with_conv_transpose(x_frames, n_fft, hop_length, window)
print(reconstructed_signal.shape)
