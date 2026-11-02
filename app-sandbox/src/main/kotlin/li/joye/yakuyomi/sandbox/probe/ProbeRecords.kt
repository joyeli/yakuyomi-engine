package li.joye.yakuyomi.sandbox.probe

/**
 * 從紀錄檔算關卡（純函式，JVM 上測得到）。紀錄檔本身就是狀態：閃退後接著跑是另一個行程，計數一律從檔案算，不靠記憶體。
 *
 * 原則：沒比到不等於相同。少了組合、缺基準、步驟失敗都不能回報「通過」。
 */
internal object ProbeRecords {
    /** 畫面離開最上層時寫的 WARN（[foreground] 靠這句認）。 */
    const val LEFT_FOREGROUND = "畫面離開最上層（切走或鎖屏），之後的時間不能算"

    /** 一行拆成鍵值（第一欄是種類）。閃退可能留下半行：拿不到的鍵就是沒有。 */
    fun fields(line: String): Map<String, String> =
        line.split('\t').drop(1).associate { it.substringBefore('=') to it.substringAfter('=', "") }

    fun records(lines: List<String>, kind: String): List<Map<String, String>> =
        lines.filter { it.startsWith("$kind\t") }.map { fields(it) }

    /** 比對的結論：有不同＝false；沒有不同但沒比滿＝incomplete（不是通過）；比滿且全同＝true。 */
    fun verdict(mismatches: Int, compared: Int, want: Int): String = when {
        mismatches != 0 -> "false"
        compared < want -> "incomplete"
        else -> "true"
    }

    class Pairs(val pairs: Int, val compared: Int, val mismatches: Int)

    /** PAIR 列：同一個組合重複出現（閃退前剛寫完）只算一次；有任何一次缺基準的組合不算比到。 */
    fun pairs(lines: List<String>): Pairs {
        val seen = HashSet<String>()
        var compared = 0
        var bad = 0
        for (kv in records(lines, "PAIR")) {
            if (!seen.add("${kv["threads"]}/${kv["page"]}/${kv["a"]}+${kv["b"]}")) continue
            bad += kv["mismatches"]?.toIntOrNull() ?: 0
            val n = kv["compared"]?.toIntOrNull() ?: 0
            val none = kv["noExpect"]?.toIntOrNull() ?: 0
            if (n > 0 && none == 0) compared++
        }
        return Pairs(seen.size, compared, bad)
    }

    class Tp(val wallMs: Map<String, Long>, val comparedModes: Int, val mismatches: Int, val noExpect: Int)

    /** TP 列：一種模式只算一次；這個模式真的比到、也沒有缺基準的才算比到。 */
    fun throughput(lines: List<String>): Tp {
        val wall = LinkedHashMap<String, Long>()
        val counted = HashSet<String>()
        var compared = 0
        var bad = 0
        var none = 0
        for (kv in records(lines, "TP")) {
            val mode = kv["mode"] ?: continue
            if (!counted.add(mode)) continue
            kv["wallMs"]?.toLongOrNull()?.let { wall[mode] = it }
            bad += kv["mismatches"]?.toIntOrNull() ?: 0
            val miss = kv["noExpect"]?.toIntOrNull() ?: 0
            none += miss
            if ((kv["compared"]?.toIntOrNull() ?: 0) > 0 && miss == 0) compared++
        }
        return Tp(wall, compared, bad, none)
    }

    fun failedSteps(lines: List<String>): List<String> =
        records(lines, "FAIL").mapNotNull { it["step"] }.distinct()

    class Foreground(val left: Int, val steps: Set<String>)

    /** 離開最上層幾次、影響哪幾步（WARN 算次數；ENV 的 importance 不是 100＝那個時間點不在最上層）。 */
    fun foreground(lines: List<String>): Foreground {
        var left = 0
        val steps = LinkedHashSet<String>()
        var cur = "開始前"
        for (l in lines) {
            when {
                l.startsWith("STEP\t") -> fields(l)["id"]?.let { cur = it }
                l.startsWith("WARN\t") -> if (fields(l)["what"] == LEFT_FOREGROUND) {
                    left++
                    steps += cur
                }
                l.startsWith("ENV\t") -> {
                    val kv = fields(l)
                    val imp = kv["importance"]
                    if (imp != null && imp != "100") steps += kv["step"] ?: cur
                }
            }
        }
        return Foreground(left, steps)
    }

    /** 這份紀錄的 tombstone 檔名：同一輪可能閃退好幾次（跳過繼續跑），每次一個檔，帶步驟與時間才不會互相蓋掉。 */
    fun tombstoneName(reportFile: String, step: String?, time: String): String {
        val tag = (step ?: "outside").replace(Regex("[^A-Za-z0-9._-]"), "_")
        return "${reportFile.removeSuffix(".txt")}_crash_${tag}_$time.pb"
    }

    /** 這個檔是不是這份紀錄（或它的 tombstone）。 */
    fun belongsToReport(name: String, reportFile: String): Boolean =
        name == reportFile || (name.startsWith(reportFile.removeSuffix(".txt") + "_crash") && name.endsWith(".pb"))
}
