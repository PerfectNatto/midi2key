// play_wav_async_filter_pthreads.c

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdbool.h>

// リングバッファー構造体
typedef struct {
    float* buffer;
    size_t size;         // バッファーのサイズ（フレーム数）
    size_t read_pos;
    size_t write_pos;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    bool finished;       // デコーディングが終了したかどうか
} RingBuffer;

// リングバッファーの初期化
int ring_buffer_init(RingBuffer* rb, size_t size) {
    rb->buffer = (float*)malloc(size * sizeof(float));
    if (rb->buffer == NULL) return -1;
    rb->size = size;
    rb->read_pos = 0;
    rb->write_pos = 0;
    rb->finished = false;
    if (pthread_mutex_init(&rb->mutex, NULL) != 0) return -1;
    if (pthread_cond_init(&rb->cond, NULL) != 0) return -1;
    return 0;
}

// リングバッファーの解放
void ring_buffer_free(RingBuffer* rb) {
    free(rb->buffer);
    pthread_mutex_destroy(&rb->mutex);
    pthread_cond_destroy(&rb->cond);
}

// 書き込み関数
size_t ring_buffer_write(RingBuffer* rb, const float* data, size_t frames) {
    pthread_mutex_lock(&rb->mutex);
    size_t frames_written = 0;
    for (size_t i = 0; i < frames; i++) {
        size_t next_pos = (rb->write_pos + 1) % rb->size;
        if (next_pos == rb->read_pos) {
            break; // バッファーが満杯
        }
        rb->buffer[rb->write_pos] = data[i];
        rb->write_pos = next_pos;
        frames_written++;
    }
    if (frames_written > 0) {
        pthread_cond_signal(&rb->cond); // 読み取りスレッドを通知
    }
    pthread_mutex_unlock(&rb->mutex);
    return frames_written;
}

// 読み取り関数
size_t ring_buffer_read(RingBuffer* rb, float* data, size_t frames) {
    pthread_mutex_lock(&rb->mutex);
    size_t frames_read = 0;
    while (rb->read_pos == rb->write_pos && !rb->finished) {
        pthread_cond_wait(&rb->cond, &rb->mutex); // データが入るまで待機
    }
    for (size_t i = 0; i < frames; i++) {
        if (rb->read_pos == rb->write_pos) {
            break; // バッファーが空
        }
        data[i] = rb->buffer[rb->read_pos];
        rb->read_pos = (rb->read_pos + 1) % rb->size;
        frames_read++;
    }
    pthread_mutex_unlock(&rb->mutex);
    return frames_read;
}

// フィルタ処理（音量調整の例）
void apply_filter(float* data, size_t frames, size_t channels) {
    float volume = 0.5f; // 音量を半分にする
    for (size_t i = 0; i < frames * channels; i++) {
        data[i] *= volume;
    }
}

// フィルタスレッドのデータ構造
typedef struct {
    RingBuffer* input_rb;
    RingBuffer* output_rb;
    size_t frames_per_block;
    size_t channels;
    bool running;
} FilterThreadData;

// フィルタスレッドの実装
void* filter_thread_func(void* arg) {
    FilterThreadData* ft = (FilterThreadData*)arg;
    float* buffer = (float*)malloc(ft->frames_per_block * ft->channels * sizeof(float));
    if (buffer == NULL) {
        fprintf(stderr, "メモリ割り当て失敗\n");
        return NULL;
    }

    while (ft->running) {
        size_t frames_read = ring_buffer_read(ft->input_rb, buffer, ft->frames_per_block * ft->channels);
        if (frames_read > 0) {
            apply_filter(buffer, frames_read, ft->channels);
            ring_buffer_write(ft->output_rb, buffer, frames_read * ft->channels);
        }
        if (ft->input_rb->finished && ring_buffer_read(ft->input_rb, buffer, ft->frames_per_block * ft->channels) == 0) {
            break;
        }
    }

    free(buffer);
    return NULL;
}

// データコールバック関数
void data_callback(ma_device* device, void* output, const void* input, ma_uint32 frame_count) {
    RingBuffer* output_rb = (RingBuffer*)device->pUserData;
    float* out = (float*)output;

    size_t frames_needed = frame_count * device->playback.channels;
    size_t frames_read = ring_buffer_read(output_rb, out, frames_needed);

    // 必要なフレーム数に満たない場合、残りをゼロクリア
    if (frames_read < frames_needed) {
        memset(out + frames_read, 0, (frames_needed - frames_read) * sizeof(float));
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("使用法: %s <wavファイル>\n", argv[0]);
        return -1;
    }

    const char* filename = argv[1];
    ma_result result;
    ma_decoder decoder;
    ma_device device;

    // 入力と出力のリングバッファーを初期化
    RingBuffer input_rb;
    RingBuffer output_rb;
    size_t buffer_size = 44100 * 10; // 10秒分のフレーム（例）
    if (ring_buffer_init(&input_rb, buffer_size) != 0 ||
        ring_buffer_init(&output_rb, buffer_size) != 0) {
        fprintf(stderr, "リングバッファーの初期化に失敗しました\n");
        return -1;
    }

    // デコーダーの初期化（float フォーマットでデコード）
    ma_decoder_config decoderConfig = ma_decoder_config_init(ma_format_f32, 0, 0);
    result = ma_decoder_init_file(filename, &decoderConfig, &decoder);
    if (result != MA_SUCCESS) {
        printf("ファイルを開けませんでした: %s\n", filename);
        ring_buffer_free(&input_rb);
        ring_buffer_free(&output_rb);
        return -1;
    }

    size_t channels = decoder.outputChannels;
    size_t sampleRate = decoder.outputSampleRate;

    // フィルタスレッドの設定
    FilterThreadData ft_data;
    ft_data.input_rb = &input_rb;
    ft_data.output_rb = &output_rb;
    ft_data.frames_per_block = 1024; // 処理ブロックサイズ
    ft_data.channels = channels;
    ft_data.running = true;

    pthread_t filter_thread;
    if (pthread_create(&filter_thread, NULL, filter_thread_func, &ft_data) != 0) {
        fprintf(stderr, "フィルタスレッドの作成に失敗しました\n");
        ma_decoder_uninit(&decoder);
        ring_buffer_free(&input_rb);
        ring_buffer_free(&output_rb);
        return -1;
    }

    // デバイス設定の初期化
    ma_device_config deviceConfig = ma_device_config_init(ma_device_type_playback);
    deviceConfig.playback.format   = ma_format_f32;
    deviceConfig.playback.channels = channels;
    deviceConfig.sampleRate        = sampleRate;
    deviceConfig.dataCallback      = data_callback;
    deviceConfig.pUserData         = &output_rb;

    // デバイスの初期化
    if (ma_device_init(NULL, &deviceConfig, &device) != MA_SUCCESS) {
        printf("デバイスを初期化できませんでした.\n");
        ft_data.running = false;
        pthread_join(filter_thread, NULL);
        ma_decoder_uninit(&decoder);
        ring_buffer_free(&input_rb);
        ring_buffer_free(&output_rb);
        return -1;
    }

    // 再生開始
    if (ma_device_start(&device) != MA_SUCCESS) {
        printf("デバイスを開始できませんでした.\n");
        ma_device_uninit(&device);
        ft_data.running = false;
        pthread_join(filter_thread, NULL);
        ma_decoder_uninit(&decoder);
        ring_buffer_free(&input_rb);
        ring_buffer_free(&output_rb);
        return -1;
    }

    printf("再生中: %s\n", filename);
    printf("終了するには Enter キーを押してください...\n");

    // メインスレッドでデータをデコードして入力バッファーに書き込む
    float* decode_buffer = (float*)malloc(ft_data.frames_per_block * channels * sizeof(float));
    if (decode_buffer == NULL) {
        fprintf(stderr, "デコードバッファーの割り当てに失敗しました\n");
        ft_data.running = false;
        pthread_join(filter_thread, NULL);
        ma_device_uninit(&device);
        ma_decoder_uninit(&decoder);
        ring_buffer_free(&input_rb);
        ring_buffer_free(&output_rb);
        return -1;
    }

    while (1) {
        ma_uint64 frames_read = ma_decoder_read_pcm_frames(&decoder, decode_buffer, ft_data.frames_per_block * channels, NULL);
        if (frames_read == 0) {
            // デコード終了
            pthread_mutex_lock(&input_rb.mutex);
            input_rb.finished = true;
            pthread_cond_signal(&input_rb.cond);
            pthread_mutex_unlock(&input_rb.mutex);
            break;
        }
        ring_buffer_write(&input_rb, decode_buffer, frames_read * channels);
    }

    free(decode_buffer);

    // 再生終了を待つ
    getchar();

    // 終了処理
    ft_data.running = false;
    pthread_join(filter_thread, NULL);
    ma_device_uninit(&device);
    ma_decoder_uninit(&decoder);
    ring_buffer_free(&input_rb);
    ring_buffer_free(&output_rb);

    return 0;
}
