#!/usr/bin/env bash
# 引擎原生庫的守門（CI 與本機共用）：
#   1. APK 每個 ABI 只有一個 .so：libyakuyomi_ncnn.so（ncnn 是靜態庫，兩個 .so 各帶一份就有兩個 SimpleOMP 常駐池）。
#   2. llvm-nm -D 只看得到 JNI（JNI_OnLoad*、Java_*）；夜讀的 JNI 入口都在（握手、第 2 階與第 3a 階原語各抽一個）。
#   3. 夜讀的物件檔沒有 FMA、沒有 OpenMP 參照；反面自檢：同一支掃描要抓得到 NDK 預設旗標編出來的 fmadd。
#   4. 夜讀的編譯旗標守住逐位元規則（最後的 -O 是 -O2、-ffp-contract=off…；nightread 的 check_compile_flags.py）。
#
# 用法：check_native_so.sh <apk> <cxx 變體目錄，例如 inference-core/.cxx/RelWithDebInfo> [--probe]
#   --probe：這是探針的建置，要有 NativeTestBridge 的 JNI；沒給＝產品的建置，一個都不能有。
# NDK：ANDROID_NDK_ROOT／ANDROID_NDK_HOME，否則 $ANDROID_HOME/ndk/<版本>；WSL 上也吃 Windows 版工具（.exe）。
set -euo pipefail
apk="${1:?apk}"
cxx="${2:?cxx 變體目錄}"
probe="${3:-}"
NDK_VER=28.2.13676358
root="$(cd "$(dirname "$0")/../.." && pwd)"
fail=0

ndk="${ANDROID_NDK_ROOT:-${ANDROID_NDK_HOME:-}}"
if [ -z "$ndk" ]; then
  sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
  [ -z "$sdk" ] && [ -d /mnt/c/Users ] && sdk="$(ls -d /mnt/c/Users/*/AppData/Local/Android/Sdk 2>/dev/null | head -1)"
  ndk="$sdk/ndk/$NDK_VER"
fi
bin="$(ls -d "$ndk"/toolchains/llvm/prebuilt/*/bin | head -1)"
tool() { if [ -x "$bin/$1" ]; then echo "$bin/$1"; else echo "$bin/$1.exe"; fi; }
nm="$(tool llvm-nm)"; objdump="$(tool llvm-objdump)"; clang="$(tool clang++)"

work="$(mktemp -d "$cxx/check.XXXXXX")"   # 放在 cxx 目錄底下：Windows 版工具讀不到 WSL 的 /tmp
trap 'rm -rf "${work:?}"' EXIT

# 1
listing="$(python3 - "$apk" <<'PY'
import sys, zipfile
for n in zipfile.ZipFile(sys.argv[1]).namelist():
    if n.startswith("lib/") and n.endswith(".so"):
        print(n)
PY
)"
echo "APK 的 .so："; echo "$listing" | sed 's/^/  /'
abis="$(echo "$listing" | cut -d/ -f2 | sort -u)"
[ -n "$abis" ] || { echo "APK 裡沒有 .so"; fail=1; }
for a in $abis; do
  got="$(echo "$listing" | grep "^lib/$a/")"
  if [ "$got" != "lib/$a/libyakuyomi_ncnn.so" ]; then echo "ABI $a 不是只有 libyakuyomi_ncnn.so：$got"; fail=1; fi
done

# 2
python3 - "$apk" "$work" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
for n in z.namelist():
    if n.startswith("lib/arm64-v8a/") and n.endswith(".so"):
        open(sys.argv[2] + "/apk.so", "wb").write(z.read(n))
PY
syms="$(cd "$work" && "$nm" -D --defined-only apk.so | awk '{print $NF}' | tr -d '\r')"
extra="$(echo "$syms" | grep -vE '^(Java_|JNI_OnLoad)' || true)"
if [ -n "$extra" ]; then echo "多匯出的符號（$(echo "$extra" | wc -l) 個）："; echo "$extra" | head -20; fail=1; fi
need="Java_li_joye_yakuyomi_engine_NcnnBackend_extractNative Java_li_joye_yakuyomi_nightread_NightNative_abi Java_li_joye_yakuyomi_nightread_NightNativePrims_nDilatePacked Java_li_joye_yakuyomi_nightread_NightNativeBin_nMask"
for s in $need; do
  echo "$syms" | grep -qx "$s" || { echo "缺 JNI 符號：$s"; fail=1; }
done
nbridge="$(echo "$syms" | grep -c 'NativeTestBridge' || true)"
if [ "$probe" = "--probe" ]; then
  [ "$nbridge" -gt 0 ] || { echo "探針的 .so 沒有 NativeTestBridge"; fail=1; }
else
  [ "$nbridge" = 0 ] || { echo "產品的 .so 不該有 NativeTestBridge（$nbridge 個）"; fail=1; }
fi
echo "匯出：$(echo "$syms" | wc -l) 個（NativeTestBridge $nbridge），其他 $(printf '%s' "$extra" | grep -c . || true) 個"

# 3（變體底下可能有舊的 hash 目錄：只看跟 APK 這顆 .so 同一次建置的，由 build-id 找）
readelf="$(tool llvm-readelf)"
id_of() { "$readelf" -n "$1" | grep 'Build ID:' | awk '{print $NF}' | tr -d '\r'; }
apk_id="$(cd "$work" && id_of apk.so)"
build=""
for d in "$cxx"/*/arm64-v8a; do
  [ -d "$d/nightread_native" ] || continue
  so="$(cd "$(dirname "$cxx")/../build/intermediates/cxx/$(basename "$cxx")/$(basename "$(dirname "$d")")/obj/arm64-v8a" 2>/dev/null && pwd)/libyakuyomi_ncnn.so"
  [ -f "$so" ] || continue
  if [ "$(cd "$(dirname "$so")" && id_of libyakuyomi_ncnn.so)" = "$apk_id" ]; then build="$d"; fi
done
[ -n "$build" ] || { echo "找不到跟 APK 同一個 build-id（$apk_id）的建置目錄：$cxx"; exit 2; }
echo "建置目錄：$build（build-id $apk_id）"
mapfile -t objs < <(cd "$build" && find nightread_native/CMakeFiles -name '*.o')
n=0
for o in "${objs[@]}"; do
  hits="$(cd "$build" && "$objdump" -d --no-show-raw-insn "$o" | grep -Ei $'\t(fmadd|fmsub|fnmadd|fnmsub|fmla|fmls)\\b' || true)"
  if [ -n "$hits" ]; then echo "FMA：$o"; echo "$hits" | head -5; fail=1; fi
  omp="$(cd "$build" && "$nm" "$o" | grep -E '__kmpc|omp_' || true)"
  if [ -n "$omp" ]; then echo "OpenMP 參照：$o"; echo "$omp" | head -5; fail=1; fi
  n=$((n + 1))
done
[ "$n" -ge 20 ] || { echo "夜讀物件檔太少（$n 個）：路徑不對？"; fail=1; }
printf 'double f(double a,double b,double c){return a*b+c;}\n' > "$work/p.cpp"
# 反組譯先存起來再 grep：pipefail 下 grep -q 一找到就關管子，objdump 還在寫就會收到 SIGPIPE、整條回非 0（WSL 叫 Windows 版 .exe 時偶爾發生）
selftest="$(cd "$work" && "$clang" --target=aarch64-linux-android26 -O2 -c p.cpp -o p.o && "$objdump" -d p.o)" || selftest=""
grep -qEi $'\t(fmadd)' <<<"$selftest" \
  || { echo "掃描自檢失敗：預設旗標的 a*b+c 沒反組譯出 fmadd"; fail=1; }
echo "FMA／OpenMP：掃了 $n 個夜讀物件檔"

# 4
python3 "$root/yakuyomi-nightread/native/tools/check_compile_flags.py" "$build" || fail=1

[ "$fail" = 0 ] && echo "check_native_so：通過" || echo "check_native_so：有問題"
exit "$fail"
