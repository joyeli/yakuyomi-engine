package li.joye.yakuyomi.sandbox.probe

import java.io.ByteArrayOutputStream
import java.util.zip.Inflater

/**
 * 8 位元灰階、不交錯 PNG 的最小解碼器，純 Kotlin。守門摘要的 fixture 都是這個格式。
 *
 * 不用 BitmapFactory：解碼器走色彩管理、解出來的是 ARGB，要是差一個灰階值，摘要對不上時就分不清是解碼器還是重繪。這裡逐位元就是檔案裡的樣本，
 * 跟 JVM 上 `ImageIO.read(...).raster.getSample` 相同（[GuardDigestJvmTest] 拿摘要驗）。
 */
internal object GrayPng {
    class Img(@JvmField val w: Int, @JvmField val h: Int, @JvmField val samples: IntArray)

    fun decode(b: ByteArray): Img {
        val sig = byteArrayOf(0x89.toByte(), 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A)
        for (i in sig.indices) require(b[i] == sig[i]) { "不是 PNG" }
        fun int32(o: Int) = ((b[o].toInt() and 255) shl 24) or ((b[o + 1].toInt() and 255) shl 16) or
            ((b[o + 2].toInt() and 255) shl 8) or (b[o + 3].toInt() and 255)
        var p = 8
        var w = 0
        var h = 0
        val idat = ByteArrayOutputStream(b.size)
        while (p < b.size) {
            val len = int32(p)
            val type = String(b, p + 4, 4, Charsets.ISO_8859_1)
            val data = p + 8
            when (type) {
                "IHDR" -> {
                    w = int32(data)
                    h = int32(data + 4)
                    val depth = b[data + 8].toInt()
                    val color = b[data + 9].toInt()
                    val interlace = b[data + 12].toInt()
                    require(depth == 8 && color == 0 && interlace == 0) { "只支援 8 位元灰階不交錯（depth=$depth color=$color interlace=$interlace）" }
                }
                "IDAT" -> idat.write(b, data, len)
                "IEND" -> p = b.size
            }
            if (type != "IEND") p = data + len + 4
        }
        require(w > 0 && h > 0) { "PNG 沒有 IHDR" }
        val raw = ByteArray((w + 1) * h)
        val inf = Inflater()
        try {
            inf.setInput(idat.toByteArray())
            var off = 0
            while (off < raw.size) {
                val n = inf.inflate(raw, off, raw.size - off)
                if (n == 0) {
                    check(!inf.finished() && !inf.needsInput() && !inf.needsDictionary()) { "PNG 資料不足：$off/${raw.size}" }
                }
                off += n
            }
        } finally {
            inf.end()
        }
        val out = IntArray(w * h)
        for (y in 0 until h) {
            val r = y * (w + 1)
            val filter = raw[r].toInt()
            for (x in 0 until w) {
                val v = raw[r + 1 + x].toInt() and 255
                val a = if (x > 0) out[y * w + x - 1] else 0
                val u = if (y > 0) out[(y - 1) * w + x] else 0
                val c = if (x > 0 && y > 0) out[(y - 1) * w + x - 1] else 0
                val pred = when (filter) {
                    0 -> 0
                    1 -> a
                    2 -> u
                    3 -> (a + u) / 2
                    4 -> {
                        val q = a + u - c
                        val pa = Math.abs(q - a)
                        val pb = Math.abs(q - u)
                        val pc = Math.abs(q - c)
                        if (pa <= pb && pa <= pc) a else if (pb <= pc) u else c
                    }
                    else -> error("未知的 PNG 濾波 $filter")
                }
                out[y * w + x] = (v + pred) and 255
            }
        }
        return Img(w, h, out)
    }
}
