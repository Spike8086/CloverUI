#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "llama.h"
#include <algorithm>
#include <algorithm>
#include <atomic> // 引入原子锁
#include <chrono>

static std::atomic<bool> cancel_generation(false); // 全局中断标志

#define TAG "CloverLlama"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static struct llama_model * model = nullptr;
static struct llama_context * ctx = nullptr;
static struct llama_sampler * smpl = nullptr;
static const struct llama_vocab * vocab = nullptr;

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_app_LlamaPlugin_nativeLoadModel(JNIEnv *env, jobject thiz, jstring path) {
    const char *native_path = env->GetStringUTFChars(path, 0);
    std::string model_path = native_path;
    env->ReleaseStringUTFChars(path, native_path);

    LOGI("開始加載模型...");

    if (ctx) { llama_free(ctx); ctx = nullptr; }
    if (model) { llama_model_free(model); model = nullptr; }
    if (smpl) { llama_sampler_free(smpl); smpl = nullptr; }

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    // 【核心修復 1】：強制禁用 GPU (Vulkan)，徹底避開 Adreno 驅動的 Shader 編譯崩潰
    mparams.n_gpu_layers = 0;
    mparams.use_mmap = false;
    model = llama_model_load_from_file(model_path.c_str(), mparams);

    if (!model) return env->NewStringUTF("Error: 模型讀取失敗");

    vocab = llama_model_get_vocab(model);
    // ════════ 新增：获取模型架构类型 ════════
    char arch_buf[128];
    // 读取 gguf 文件中的 general.architecture 属性 (通常是 qwen2, gemma, llama 等)
    int32_t res = llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof(arch_buf));
    std::string arch_str = "Unknown";
    if (res >= 0) {
            arch_str = arch_buf;
    }
    LOGI("检测到模型架构: %s", arch_str.c_str());

    // 将架构信息拼接在 Success 后面返回给 Java，例如 "Success|qwen2"
    std::string ret_msg = "Success|" + arch_str;

    LOGI("模型加載成功！");
    return env->NewStringUTF(ret_msg.c_str());
}

// 供 Java 调用的停止方法
extern "C" JNIEXPORT void JNICALL
Java_com_example_app_LlamaPlugin_nativeStop(JNIEnv *env, jobject thiz) {
    cancel_generation = true;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_com_example_app_LlamaPlugin_nativeGenerate(JNIEnv *env, jobject thiz, jstring prompt, jint max_tokens, jstring system_prompt, jint context_size, jint threads) {
    if (!model || !vocab) return env->NewByteArray(0);

    const char *native_prompt = env->GetStringUTFChars(prompt, 0);
    const char *native_sys = env->GetStringUTFChars(system_prompt, 0);

    // ════════ 核心修改 ════════
    std::string full_prompt = std::string(native_prompt);

    // 释放内存
    env->ReleaseStringUTFChars(prompt, native_prompt);
    env->ReleaseStringUTFChars(system_prompt, native_sys);
    // ════════ 修改结束 ════════

    // 稍微增加一点预留空间 +16，防止溢出
    std::vector<llama_token> tokens_list(full_prompt.length() + 16);
    int n_tokens = llama_tokenize(vocab, full_prompt.c_str(), (int)full_prompt.length(), tokens_list.data(), (int)tokens_list.size(), true, true);
    if (n_tokens < 0) {
        tokens_list.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, full_prompt.c_str(), (int)full_prompt.length(), tokens_list.data(), (int)tokens_list.size(), true, true);
    }

    if (ctx) { llama_free(ctx); }
    struct llama_context_params cparams = llama_context_default_params();

    // === 修改核心：使用传入的 context_size ===
    // 如果传入值小于等于0，或者小于生成的 prompt token 数，强制设为 2048 或更大
    if (context_size <= 0) context_size = 1024;

    // 自动调整：如果用户设置的上下文比当前的 prompt 还短，那就太尴尬了，自动扩容
    if (context_size < n_tokens + max_tokens) {
        context_size = n_tokens + max_tokens + 128; // 稍微多留点
        LOGI("Auto-adjusting context size to %d", context_size);
    }

    cparams.n_ctx = context_size; // 应用设置
    // 陷阱 1 修复 (上)：將單次計算批次限制在手機甜點區間 (512)
    cparams.n_batch = 512;
    // ======================================
    int safe_threads = (threads > 0) ? threads : 4;

    cparams.n_threads = safe_threads;
    cparams.n_threads_batch = safe_threads;
    ctx = llama_init_from_model(model, cparams);

    if (!ctx) return env->NewByteArray(0);

    if (smpl) { llama_sampler_free(smpl); }
    smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    uint32_t random_seed = (uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(random_seed));

    // ════════ 陷阱 1 修复 (下)：将 Prompt 切块喂给模型，避免瞬间炸显存/内存 ════════
    int n_batch_size = 512; // 每次最多只吃 512 個 Token

    // 掐表 1：记录 Prompt 解码速度
    auto t_prompt_start = std::chrono::high_resolution_clock::now();

    int n_eval = 0;
    while (n_eval < n_tokens) {
        // 計算這一塊要處理多少個 token
        int chunk_size = std::min(n_batch_size, n_tokens - n_eval);
        struct llama_batch chunk_batch = llama_batch_init(chunk_size, 0, 1);
        chunk_batch.n_tokens = chunk_size;

        for (int i = 0; i < chunk_size; i++) {
            chunk_batch.token[i] = tokens_list[n_eval + i];
            chunk_batch.pos[i] = n_eval + i;
            chunk_batch.n_seq_id[i] = 1;
            chunk_batch.seq_id[i][0] = 0;
            chunk_batch.logits[i] = false;
        }

        // 只有整個 Prompt 的最後一個 token 才需要告訴模型輸出概率 (Logits)
        if (n_eval + chunk_size == n_tokens) {
            chunk_batch.logits[chunk_size - 1] = true;
        }

        // 分塊進行 decode
        if (llama_decode(ctx, chunk_batch) != 0) {
            llama_batch_free(chunk_batch);
            return env->NewByteArray(0);
        }

        // 計算完這一塊就釋放記憶體，防止記憶體狂飆
        llama_batch_free(chunk_batch);
        n_eval += chunk_size;
    }

    auto t_prompt_end = std::chrono::high_resolution_clock::now();
    double prompt_ms = std::chrono::duration<double, std::milli>(t_prompt_end - t_prompt_start).count();
    double prompt_speed = (prompt_ms > 0) ? (n_tokens / (prompt_ms / 1000.0)) : 0.0;

    // ════════ 準備進入第二階段：單字生成 ════════
    // 為後面的單字生成 (Token generation) 準備一個容量為 1 的專用 batch
    struct llama_batch batch = llama_batch_init(1, 0, 1);

    std::string result = "";
    int n_cur = n_tokens;
    int n_decode = max_tokens > 0 ? max_tokens : 256;
    cancel_generation = false;

    // 获取 Java 层的 onTokenGenerated 方法
    jclass clazz = env->GetObjectClass(thiz);
    jmethodID onTokenMethod = env->GetMethodID(clazz, "onTokenGenerated", "([B)V");
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
        onTokenMethod = nullptr;
    }

    // ════════ 掐表 2：记录文字生成速度 ════════
    auto t_gen_start = std::chrono::high_resolution_clock::now();
    int gen_count = 0; // 记录真实吐了多少个 Token

    while (n_cur <= n_tokens + n_decode) {
        // 检查全局取消标志（对应 nativeStop）
        if (cancel_generation) break;

        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_accept(smpl, id);

        if (llama_vocab_is_eog(vocab, id)) break;

        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string new_piece(buf, n);
            result += new_piece; // C++ 依然自己记录完整结果用于最后返回
            // 修复 3：只把这个"新字"传给 Java，别传整个句子
            if (onTokenMethod != nullptr) {
                jbyteArray jbytes = env->NewByteArray((jsize)new_piece.length());
                env->SetByteArrayRegion(jbytes, 0, (jsize)new_piece.length(), (const jbyte*)new_piece.c_str());
                env->CallVoidMethod(thiz, onTokenMethod, jbytes);
                env->DeleteLocalRef(jbytes);
            }
        }

        batch.n_tokens = 1;
        batch.token[0] = id;
        batch.pos[0] = n_cur;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;

        if (llama_decode(ctx, batch) != 0) break;
        n_cur++;
        gen_count++;
    }

    auto t_gen_end = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t_gen_end - t_gen_start).count();
    double gen_speed = (gen_ms > 0) ? (gen_count / (gen_ms / 1000.0)) : 0.0;
    // ════════ 把测速结果伪装成暗号拼在最后面 ════════
    char stats_buf[128];
    // 格式例如：[CLOVER_STATS|24.50|8.32]
    snprintf(stats_buf, sizeof(stats_buf), "[CLOVER_STATS|%.2f|%.2f]", prompt_speed, gen_speed);
    result += std::string(stats_buf);

    llama_batch_free(batch);
    env->DeleteLocalRef(clazz);

    jbyteArray ret = env->NewByteArray((jsize)result.length());
    env->SetByteArrayRegion(ret, 0, (jsize)result.length(), (const jbyte*)result.c_str());
    return ret;
}