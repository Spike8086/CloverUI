package com.example.app;

import android.content.Intent;
import android.net.Uri;
import androidx.activity.result.ActivityResult;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.ActivityCallback;
import com.getcapacitor.annotation.CapacitorPlugin;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import android.database.Cursor;
import android.provider.OpenableColumns;
import android.os.Environment;
import android.provider.Settings;

@CapacitorPlugin(name = "Llama")
public class LlamaPlugin extends Plugin {
    // 1. 必须在类级别定义这个变量，否则方法里找不到它
    private boolean isGenerating = false;

    static {
        System.loadLibrary("clover-bridge");
    }

    public native String nativeLoadModel(String path);

    // 确保这里是 byte[]，对应 C++ 的安全返回
    public native byte[] nativeGenerate(String prompt, int maxTokens, String systemPrompt, int contextSize, int threads);

    public native void nativeStop(); // 声明 C++ 的停止方法

    // 暴露给前端 JS 的中断接口
    @PluginMethod
    public void stop(PluginCall call) {
        nativeStop();
        isGenerating = false;
        call.resolve();
    }

    // 【修改版】接收 C++ 的流式数据，并强制在 UI 主线程推送给网页
    public void onTokenGenerated(byte[] bytes) {
        try {
            // 解析字节流
            String text = new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
            JSObject ret = new JSObject();
            ret.put("text", text);

            // 【核心修复】：必须强制在安卓主线程发送事件，前端网页才会立刻产生打字机视觉效果！
            if (getActivity() != null) {
                getActivity().runOnUiThread(() -> {
                    notifyListeners("onToken", ret);
                });
            }
        } catch (Exception e) {
            // 忽略由于半个汉字导致的临时解析异常
        }
    }
    // 核心修復 1：放入子線程，防止切換模型時 App 假死
    @PluginMethod
    public void load(PluginCall call) {
        new Thread(() -> {
            android.os.ParcelFileDescriptor pfd = null;
            try {
                String path = call.getString("path");
                String realPathToLoad = path;

                // 🌟 核心魔法 3：如果是外部导入的 content:// 文件
                if (path != null && path.startsWith("content://")) {
                    Uri uri = Uri.parse(path);
                    // 1. 在 Java 层打开这个外部文件，拿到系统的底层句柄 (File Descriptor)
                    pfd = getContext().getContentResolver().openFileDescriptor(uri, "r");
                    if (pfd != null) {
                        int fd = pfd.getFd();
                        // 2. 利用 Linux 万物皆文件的特性，把句柄伪装成绝对路径喂给 C++！
                        realPathToLoad = "/proc/self/fd/" + fd;
                    } else {
                        call.reject("无法打开外部文件描述符");
                        return;
                    }
                } else {
                    // 兼容你以前复制到私有目录的旧模型
                    File f = new File(path);
                    if (!f.exists()) {
                        call.reject("实体文件不存在，可能已被删除");
                        return;
                    }
                }

                String result = nativeLoadModel(realPathToLoad);

                // C++ 的 fopen 已经成功复制了句柄，Java 这边的原始句柄可以安全关闭了
                if (pfd != null) {
                    pfd.close();
                }

                if (result.startsWith("Error")) {
                    call.reject(result);
                } else {
                    JSObject ret = new JSObject();
                    ret.put("status", result);
                    call.resolve(ret);
                }
            } catch (Exception e) {
                call.reject("载入失败: " + e.getMessage());
                if (pfd != null) {
                    try { pfd.close(); } catch (Exception ignored) {}
                }
            }
        }).start();
    }

    @PluginMethod
    public void checkStoragePermission(PluginCall call) {
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            // 检查是否已拥有最高文件管理权限
            if (!Environment.isExternalStorageManager()) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION);
                intent.setData(Uri.parse("package:" + getContext().getPackageName()));
                // 跳转到系统设置页，让用户亲自打开开关
                getActivity().startActivity(intent);

                JSObject ret = new JSObject();
                ret.put("granted", false);
                call.resolve(ret);
                return;
            }
        }
        // 已经有权限，直接放行
        JSObject ret = new JSObject();
        ret.put("granted", true);
        call.resolve(ret);
    }
    // 核心修復 2：提供給前端真正刪除幾十 GB 實體檔案的能力
    @PluginMethod
    public void deleteFile(PluginCall call) {
        String path = call.getString("path");
        if (path != null) {
            if (path.startsWith("content://")) {
                // 如果是外部文件，不要去删实体文件！只释放我们申请的读取权限即可。
                try {
                    getContext().getContentResolver().releasePersistableUriPermission(Uri.parse(path), Intent.FLAG_GRANT_READ_URI_PERMISSION);
                } catch (Exception ignored) {}
            } else {
                // 只有旧的、被复制进私有目录的文件，才执行物理删除释放空间
                File file = new File(path);
                if (file.exists()) {
                    file.delete();
                }
            }
        }
        call.resolve();
    }

    @PluginMethod
    public void generate(PluginCall call) {
        if (isGenerating) {
            call.reject("AI 正在加载中...");
            return;
        }

        String prompt = call.getString("prompt");
        // 【修改点 2】从 JS 获取 maxTokens 参数，默认给 256
        Integer maxTokens = call.getInt("maxTokens", 256);

        String systemPrompt = call.getString("systemPrompt", "");
        Integer contextSize = call.getInt("contextSize", 1024);
        Integer threads = call.getInt("threads", 4);
        isGenerating = true;

        new Thread(() -> {
            try {
                // 【修改点 3】将 maxTokens 传给 C++
                byte[] bytes = nativeGenerate(prompt, maxTokens, systemPrompt, contextSize, threads);

                if (bytes == null || bytes.length == 0) {
                    call.reject("AI 生成了空结果或崩潰");
                    return;
                }

                String response = new String(bytes, java.nio.charset.StandardCharsets.UTF_8);
                String pSpeed = "0.0";
                String gSpeed = "0.0";

                // ════════ 拦截 C++ 发来的测速暗号 ════════
                String tokenMarker = "[CLOVER_STATS|";
                int markerIdx = response.lastIndexOf(tokenMarker);
                if (markerIdx != -1) {
                    int endIdx = response.lastIndexOf("]");
                    if (endIdx > markerIdx) {
                        // 提取出 24.50|8.32
                        String statsStr = response.substring(markerIdx + tokenMarker.length(), endIdx);
                        String[] parts = statsStr.split("\\|");
                        if (parts.length == 2) {
                            pSpeed = parts[0];
                            gSpeed = parts[1];
                        }
                        // 切掉暗号，保证用户看到的文本是干净的
                        response = response.substring(0, markerIdx);
                    }
                }

                JSObject ret = new JSObject();
                ret.put("content", response);
                ret.put("promptSpeed", pSpeed); // 传给前端 JS
                ret.put("genSpeed", gSpeed);    // 传给前端 JS
                call.resolve(ret);
            } catch (Exception e) {
                call.reject("發生異常: " + e.getMessage());
            } finally {
                isGenerating = false;
            }
        }).start();
    }

    @PluginMethod
    public void pickModel(PluginCall call) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        startActivityForResult(call, intent, "pickModelResult");
    }
    @ActivityCallback
    private void pickModelResult(PluginCall call, ActivityResult result) {
        if (call == null) return;

        if (result.getResultCode() == getActivity().RESULT_OK) {
            Intent data = result.getData();
            if (data != null) {
                Uri uri = data.getData();
                try {
                    String originalName = "未知文件.gguf";
                    long sizeBytes = 0;
                    Cursor cursor = getContext().getContentResolver().query(uri, null, null, null, null);
                    if (cursor != null && cursor.moveToFirst()) {
                        int nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                        int sizeIndex = cursor.getColumnIndex(OpenableColumns.SIZE);
                        if (nameIndex != -1) originalName = cursor.getString(nameIndex);
                        if (sizeIndex != -1) sizeBytes = cursor.getLong(sizeIndex);
                        cursor.close();
                    }
                    String sizeStr = String.format("%.2f GB", sizeBytes / 1073741824.0);

                    // 🌟 获取前端传来的模式："copy" (内部) 或 "link" (外部)
                    String mode = call.getString("mode", "link");

                    if ("copy".equals(mode)) {
                        // 【模式一：复制到内部沙盒】
                        final String fOriginalName = originalName;
                        final String fSizeStr = sizeStr;
                        // 开启后台线程复制文件，防止阻塞主界面
                        new Thread(() -> {
                            try {
                                InputStream is = getContext().getContentResolver().openInputStream(uri);
                                File outFile = new File(getContext().getFilesDir(), fOriginalName);
                                FileOutputStream fos = new FileOutputStream(outFile);
                                byte[] buffer = new byte[8192];
                                int length;
                                while ((length = is.read(buffer)) > 0) {
                                    fos.write(buffer, 0, length);
                                }
                                fos.close();
                                is.close();

                                JSObject ret = new JSObject();
                                ret.put("path", outFile.getAbsolutePath());
                                ret.put("size", fSizeStr);
                                ret.put("originalName", fOriginalName);
                                ret.put("storageType", "internal"); // 打上内部标签
                                call.resolve(ret);
                            } catch (Exception e) {
                                call.reject("复制文件失败: " + e.getMessage());
                            }
                        }).start();

                    } else {
                        // 【模式二：0秒外部链接直读】
                        final int takeFlags = Intent.FLAG_GRANT_READ_URI_PERMISSION;
                        getContext().getContentResolver().takePersistableUriPermission(uri, takeFlags);

                        JSObject ret = new JSObject();
                        ret.put("path", uri.toString());
                        ret.put("size", sizeStr);
                        ret.put("originalName", originalName);
                        ret.put("storageType", "external"); // 打上外部标签
                        call.resolve(ret);
                    }
                } catch (Exception e) {
                    call.reject("处理文件失败: " + e.getMessage());
                }
            } else {
                call.reject("未选择文件");
            }
        } else {
            call.reject("取消选择");
        }
    }
}
