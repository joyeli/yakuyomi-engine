package li.joye.yakuyomi.sandbox.probe

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ApplicationInfo
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.Process
import java.io.File
import java.util.zip.ZipFile

/** 裝置、建置、溫度與「app 是否還在最上層」：每一步都記，事後才知道數字是在什麼條件下量的。 */
class ProbeEnv(private val ctx: Context) {
    private val pm = ctx.getSystemService(PowerManager::class.java)
    private val am = ctx.getSystemService(ActivityManager::class.java)

    fun debuggable(): Boolean = (ctx.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0

    /** 這條執行緒能跑的核心（`/proc/thread-self/status`）。這台手機不在最上層時只給 5 顆（0-2,5-6）。 */
    fun cpusAllowed(): String = runCatching {
        File("/proc/thread-self/status").readLines().first { it.startsWith("Cpus_allowed_list") }.substringAfter(':').trim()
    }.getOrDefault("?")

    /** 所有核心的最高頻率（判斷大小核；讀不到就算了）。 */
    fun cpuFreqs(): String = runCatching {
        (0 until Runtime.getRuntime().availableProcessors()).joinToString(",") { i ->
            File("/sys/devices/system/cpu/cpu$i/cpufreq/cpuinfo_max_freq").readText().trim().toLong().div(1000).toString()
        }
    }.getOrDefault("?")

    fun thermalStatus(): Int = if (Build.VERSION.SDK_INT >= 29) pm.currentThermalStatus else -1

    /** 0＝無、1 輕微、2 中等、3 嚴重、4 重大、5 緊急、6 關機。 */
    fun thermalName(s: Int): String = when (s) {
        0 -> "NONE"; 1 -> "LIGHT"; 2 -> "MODERATE"; 3 -> "SEVERE"; 4 -> "CRITICAL"; 5 -> "EMERGENCY"; 6 -> "SHUTDOWN"; else -> "?"
    }

    private fun headroom(): String = if (Build.VERSION.SDK_INT >= 30) {
        runCatching { "%.3f".format(pm.getThermalHeadroom(0)) }.getOrDefault("?")
    } else {
        "?"
    }

    private fun battery(): Pair<String, String> {
        val i = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED)) ?: return "?" to "?"
        val t = i.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
        val plugged = i.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0)
        val level = i.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        return (if (t == Int.MIN_VALUE) "?" else "%.1f".format(t / 10.0)) to "$plugged,lvl=$level"
    }

    /** 行程目前的重要度：100＝最上層（IMPORTANCE_FOREGROUND）。 */
    fun importance(): Int {
        val info = ActivityManager.RunningAppProcessInfo()
        ActivityManager.getMyMemoryState(info)
        return info.importance
    }

    /** 一行環境。 */
    fun snapshot(): Array<Pair<String, Any?>> {
        val (temp, plug) = battery()
        val s = thermalStatus()
        val rt = Runtime.getRuntime()
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        return arrayOf(
            "thermal" to "$s(${thermalName(s)})",
            "headroom" to headroom(),
            "battC" to temp,
            "plugged" to plug,
            "top" to ProbeSession.activityVisible,
            "importance" to importance(),
            "cpus" to cpusAllowed(),
            "cpuMs" to Process.getElapsedCpuTime(),
            "heapMB" to (rt.totalMemory() - rt.freeMemory()) / (1 shl 20),
            "nativeMB" to Debug.getNativeHeapAllocatedSize() / (1 shl 20),
            "availMB" to mi.availMem / (1 shl 20),
        )
    }

    /** 開頭的裝置與建置資訊。 */
    fun metaLines(report: ProbeReport, buildTag: String) {
        val rt = Runtime.getRuntime()
        val pi = ctx.packageManager.getPackageInfo(ctx.packageName, 0)
        val soc = if (Build.VERSION.SDK_INT >= 31) "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}" else Build.HARDWARE
        report.line(
            "META",
            "app" to ctx.packageName, "version" to pi.versionName, "buildTag" to buildTag,
            "buildType" to li.joye.yakuyomi.sandbox.BuildConfig.BUILD_TYPE, "debuggable" to debuggable(), "abis" to Build.SUPPORTED_ABIS.joinToString(","), "is64" to Process.is64Bit(),
        )
        report.line(
            "META",
            "device" to "${Build.MANUFACTURER} ${Build.MODEL}", "android" to Build.VERSION.RELEASE, "sdk" to Build.VERSION.SDK_INT,
            "soc" to soc, "vm" to System.getProperty("java.vm.version"),
        )
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        report.line(
            "META",
            "cores" to rt.availableProcessors(), "coreMaxMHz" to cpuFreqs(), "cpus" to cpusAllowed(),
            "maxHeapMB" to rt.maxMemory() / (1 shl 20), "totalMemMB" to mi.totalMem / (1 shl 20),
        )
        // APK 裡實際有哪些 .so：「引擎只有一個 .so」
        runCatching {
            val apk = File(ctx.applicationInfo.sourceDir)
            val sos = ZipFile(apk).use { z ->
                z.entries().asSequence().filter { it.name.startsWith("lib/") && it.name.endsWith(".so") }
                    .map { "${it.name}:${it.size}" }.toList()
            }
            report.line("META", "apkBytes" to apk.length(), "so" to sos.joinToString(","))
        }
    }
}
