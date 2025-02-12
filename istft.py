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

import torch
import torch.nn.functional as F
import math

def ori_istft_vectorized(z, n_fft, hop_length, window, win_length, normalized=True, length=None, center=True):
    """
    z: ISTFT の入力テンソル。形状は [..., n_freq, frames, 2] （実部と虚部）
    n_fft: FFT サイズ（各フレームの出力長）
    hop_length: フレーム間のシフト
    window: [n_fft] の窓関数
    win_length: 窓長（通常 n_fft と同じかそれ以下）
    length: 出力信号の任意の長さ（None の場合は自動計算）
    center: STFT 生成時の center と同じ設定
    """
    # 実部・虚部に分割
    real, imag = z[..., 0], z[..., 1]  # shape: [..., n_freq, frames]
    n_freq, frames = real.shape[-2], real.shape[-1]
    
    # 逆DFT用の角度を計算
    t = torch.arange(n_fft, device=z.device).unsqueeze(0)         # [1, n_fft]
    k = torch.arange(n_freq, device=z.device).unsqueeze(1)          # [n_freq, 1]
    angles = 2 * math.pi * k * t / n_fft                           # [n_freq, n_fft]
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    
    # 各フレームごとに逆DFTを実施（各周波数成分から時間領域の n_fft サンプルを再構成）
    # real[..., None]: shape [..., n_freq, frames, 1]
    # cos_angles: [n_freq, n_fft] はブロードキャストされる
    x_frames = real.unsqueeze(-1) * cos_angles - imag.unsqueeze(-1) * sin_angles
    # 窓関数を適用（window の shape は [n_fft]）
    x_frames = x_frames * window
    
    # 通常、ISTFT では各フレームは周波数ビンごとに再構成されるが、
    # ここでは多くの場合、n_freq = n_fft//2+1 となっているので、
    # 各フレームの逆変換結果を (周波数方向) に合成する必要があります。
    # ここでは単純化のため、もし n_freq > 1 なら各フレームについて和をとる（＝位相情報を失う）か、
    # 事前にモデル側で n_freq=1（実際の時間領域フレームが得られている状態）と仮定してください。
    # ※ 必要に応じて x_frames = x_frames.sum(dim=-3) などで調整してください。
    if n_freq > 1:
        x_frames = x_frames.sum(dim=-3)  # 結果の shape: [batch, frames, n_fft]
    else:
        x_frames = x_frames.squeeze(-3)   # 結果の shape: [batch, frames, n_fft]
    
    # 出力信号長を決定
    if hop_length is None:
        hop_length = win_length // 4
    output_length = hop_length * (frames - 1) + n_fft
    if length is not None:
        output_length = length
    
    # ---- オーバーラップ・アド（Overlap-Add）を F.fold で実装 ----
    # x_frames の shape は [batch, frames, n_fft]
    # → まず、frames 軸と n_fft 軸を入れ替えて [batch, n_fft, frames]
    x_frames_perm = x_frames.transpose(1, 2)
    batch = x_frames_perm.shape[0]

    expected_blocks = (output_length - n_fft) // hop_length + 1

    # x_frames_perm の frames 次元が expected_blocks と一致していなければ、切り詰めまたはパディングします
    if frames > expected_blocks:
        # 多すぎる場合は先頭 expected_blocks 分だけ採用
        x_frames_perm = x_frames_perm[..., :expected_blocks]
    elif frames < expected_blocks:
        # 不足する場合はゼロパディング（右側に）します
        pad_size = expected_blocks - frames
        pad_tensor = torch.zeros(batch, n_fft, pad_size, device=z.device, dtype=z.dtype)
        x_frames_perm = torch.cat([x_frames_perm, pad_tensor], dim=-1)

    # reshape：F.fold の入力は [batch, C * kernel_size, L] となるので、ここでは C = 1, kernel_size = n_fft, L = expected_blocks
    x_frames_for_fold = x_frames_perm.view(batch, 1 * n_fft, expected_blocks)
    
    # F.fold は 2D 入力を扱うため、1D 信号は高さ1とみなします。
    # ここで、入力として期待される形は [batch, C * kernel_size, L] です。
    # ここでは、C = 1, kernel_size = n_fft, L = frames
    x_frames_for_fold = x_frames_perm.view(batch, 1 * n_fft, frames)
    
    # F.fold で重なり合うパッチを足し合わせて元の信号に戻す。
    # 出力サイズを (1, output_length) とし、カーネルサイズ (1, n_fft)、
    # ストライドを (1, hop_length) とします。
    x_fold = F.fold(x_frames_for_fold, output_size=(1, output_length),
                      kernel_size=(1, n_fft), stride=(1, hop_length))
    # 得られる x_fold の shape は [batch, 1, 1, output_length]
    x_reconstructed = x_fold.squeeze(2).squeeze(1)  # → [batch, output_length]
    # 指定があれば長さで切り出す
    return x_reconstructed[..., :length] if length is not None else x_reconstructed

import torch

def wiener_filter_no_6d_no_loop(
    mag_out: torch.Tensor,   # shape: [B, S, C, Fq, T]
    mix_stft: torch.Tensor,  # shape: [B, C, Fq, T, 2]
    eps: float = 1e-10
):
    """
    1) B と S をまとめて flatten して最終出力を [B*S, C, Fq, T, 2] とする
    2) 中間で 6 次元テンソル ([B, S, C, Fq, T, 2]) を作らない
    3) for 文で S をループしない
    を満たす実装例．
    
    戻り値: shape = [B*S, C, Fq, T, 2]
    """
    B, S, C, Fq, T = mag_out.shape

    # --- Step1: Power 計算と Wiener マスク作成 ---
    # power: [B, S, C, Fq, T]
    power = mag_out.square()

    # 全ソースのパワーを合計: [B, 1, C, Fq, T]
    power_sum = power.sum(dim=1, keepdim=True)

    # Wiener マスク（ソースごとのパワー / 合計）
    # shape: [B, S, C, Fq, T]
    mask = power / (power_sum + eps)

    # --- Step2: B と S を合体させて flat 化 ---
    # mask_flat: [B*S, C, Fq, T]
    mask_flat = mask.reshape(B * S, C, Fq, T)

    # --- Step3: mix_stft を「B 次元を S 回繰り返した」形にする ---
    # mix_stft: [B, C, Fq, T, 2] なので，
    # バッチ次元 b ∈ [0..B-1] を，ソース数 S 回ぶん繰り返すために
    # repeat_interleave や index_select を使う方法がある．

    # ここでは repeat_interleave の例
    # b_idx: [B*S]  例: B=2, S=3 => b_idx = [0,0,0, 1,1,1]
    b_idx = torch.arange(B, device=mix_stft.device).repeat_interleave(S)

    # mix_stft_flat: [B*S, C, Fq, T, 2]
    # インデックス選択で同じバッチ b を S 回ぶん取り出す
    mix_stft_flat = mix_stft[b_idx, ...]
    # こうすることで，mask_flat と mix_stft_flat がどちらも
    # [B*S, C, Fq, T, ...] の形状になり，ソース方向のブロードキャストが可能

    # --- Step4: マスクをかけて分離結果を得る ---
    # mask_flat:     [B*S, C, Fq, T]
    # mix_stft_flat: [B*S, C, Fq, T, 2]
    # unsqueeze(-1) で最後に 1 次元を足して掛け算 => [B*S, C, Fq, T, 2]
    separated_stft = mask_flat.unsqueeze(-1) * mix_stft_flat

    # shape = [B*S, C, Fq, T, 2]
    return separated_stft

import numpy as np

def idft_frame_no_complex(z_frame, nfft):
    """
    半スペクトルから時間領域フレームを計算する関数

    z_frame: shape (freq, 2)
             ここで freq = nfft/2 + 1 (例: 2049)
             z_frame[:,0] に実部, z_frame[:,1] に虚部が格納されている
    nfft: FFT サイズ（例: 4096）

    戻り値:
      x_frame: 長さ nfft の時間領域フレーム（実数配列）
    """
    # nfft が偶数なら、freq = nfft//2 + 1
    N = nfft
    freq = z_frame.shape[0]
    n = np.arange(N)  # n=0,...,N-1

    # DC 成分（k=0）の寄与
    X0 = z_frame[0, 0]  # 実部のみ
    # Nyquist 成分（k=N/2）の寄与（実際は理論上実数）
    X_nyq = z_frame[-1, 0]

    # x[n] の初期化
    x = np.zeros(N)

    # DC および Nyquist の寄与（それぞれ1倍）
    x += X0 * np.cos(2 * np.pi * 0 * n / N)  # cos(0) = 1
    x += X_nyq * np.cos(np.pi * n)            # cos(2π*(N/2)*n/N) = cos(π*n)

    # k = 1 から N/2 - 1 まで
    for k in range(1, N//2):
        Re = z_frame[k, 0]
        Im = z_frame[k, 1]
        # 各 n に対する cos と sin を計算
        cos_term = np.cos(2 * np.pi * k * n / N)
        sin_term = np.sin(2 * np.pi * k * n / N)
        # 寄与は2倍
        x += 2 * (Re * cos_term - Im * sin_term)
    # 正規化（1/N）
    return x / N

def istft(z, nfft=4096, hop_length=1024, window=None):
    """
    入力 z の shape は [batch, freq, frame, 2]
    freq = nfft//2 + 1, frame はフレーム数、2 は [実部, 虚部]
    
    オーバーラップ・アドにより全体の時間領域信号を再構成する。

    Parameters:
      z         : np.array, shape = [batch, freq, frame, 2]
      nfft      : FFT サイズ（例: 4096）
      hop_length: フレーム間シフト（例: 1024）
      window    : シンセシス窓（1次元配列, 長さ nfft）
                  None の場合は全て 1 とみなす

    戻り値:
      x_reconstructed: np.array, shape = [batch, output_length]
                       ここで output_length = hop_length*(frame-1) + nfft
    """
    batch, freq, num_frames, _ = z.shape
    output_length = hop_length * (num_frames - 1) + nfft

    # シンセシス窓が指定されなければ 1 の窓（矩形窓）
    if window is None:
        window = np.ones(nfft)
    
    # 出力用配列の初期化
    x_reconstructed = np.zeros((batch, output_length))

    # 各バッチ、各フレームごとに iDFT を計算してオーバーラップ・アド
    for b in range(batch):
        for t in range(num_frames):
            # z[b, :, t, :] shape = (freq, 2)
            x_frame = idft_frame_no_complex(z[b, :, t, :], nfft)
            # 窓を掛ける（分析時と同じ窓またはシンセシス窓）
            x_frame = x_frame * window
            start = t * hop_length
            x_reconstructed[b, start:start+nfft] += x_frame
    return x_reconstructed

# --- 使用例 ---
# 仮に nfft=4096, freq=2049, frame=48, batch=32 とする
nfft = 4096
freq = nfft//2 + 1  # 2049
num_frames = 48
batch = 32
hop_length = 1024  # 例として

# ダミーの STFT 結果（ランダムな値, 実際はモデル出力など）
# shape: [batch, freq, frame, 2]
z = np.random.randn(batch, freq, num_frames, 2)

# 必要に応じてシンセシス窓（ここでは Hann 窓の例）
window = np.hanning(nfft)

# iSTFT により時間領域信号を再構成
x_time = istft(z, nfft=nfft, hop_length=hop_length, window=window)
# x_time の shape は [batch, output_length]
print(x_time.shape)  # 例: (32, hop_length*(48-1)+4096)



