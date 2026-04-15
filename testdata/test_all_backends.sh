#!/bin/bash
source /opt/intel/oneapi/setvars.sh 2>/dev/null

VMAF=/usr/local/bin/vmaf
YUV=/home/kilian/dev/libvmaf_vulkan/python/test/resource/yuv
TD=/home/kilian/dev/libvmaf_vulkan/testdata

REF1="$YUV/src01_hrc00_576x324.yuv"
DIS1="$YUV/src01_hrc01_576x324.yuv"
REF2="$YUV/src01_hrc00_1920x1080_5frames.yuv"
DIS2="$YUV/src01_hrc01_1920x1080_5frames.yuv"

run_test() {
    local label="$1" ref="$2" dis="$3" w="$4" h="$5" bd="$6"
    shift 6
    local result
    result=$($VMAF -r "$ref" -d "$dis" -w "$w" -h "$h" -p 420 -b "$bd" "$@" -o /dev/stdout --json -q 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['pooled_metrics']['vmaf']['mean']:.6f}\")" 2>/dev/null)
    if [ -z "$result" ]; then
        result="FAILED/N/A"
    fi
    printf "  %-18s %s\n" "$label" "$result"
}

bench() {
    local label="$1" res="$2"
    shift 2
    local w=${res%x*} h=${res#*x}
    local ref="$TD/ref_${res}_48f.yuv"
    local dis="$TD/dis_${res}_48f.yuv"
    if [ ! -f "$ref" ]; then
        printf "  %-18s SKIP\n" "$label"
        return
    fi
    local fps
    fps=$(script -qc "$VMAF -r $ref -d $dis -w $w -h $h -p 420 -b 10 $* -o /dev/null --json" /dev/null 2>&1 \
        | tr '\r' '\n' | grep -oP '[\d.]+\s+FPS' | tail -1 | grep -oP '[\d.]+')
    if [ -z "$fps" ]; then fps="N/A"; fi
    printf "  %-18s %s fps\n" "$label" "$fps"
}

echo "============================================"
echo "  VMAF Score Validation - All Backends"
echo "============================================"
echo ""

echo "--- Test 1: 576x324, 48 frames, 8-bit ---"
run_test "CPU:"            "$REF1" "$DIS1" 576 324 8 --no_sycl --no_cuda
run_test "CUDA (RTX4090):" "$REF1" "$DIS1" 576 324 8 --no_sycl
run_test "SYCL (ArcA380):" "$REF1" "$DIS1" 576 324 8 --no_cuda
echo ""

echo "--- Test 2: 1920x1080, 5 frames, 8-bit ---"
run_test "CPU:"            "$REF2" "$DIS2" 1920 1080 8 --no_sycl --no_cuda
run_test "CUDA (RTX4090):" "$REF2" "$DIS2" 1920 1080 8 --no_sycl
run_test "SYCL (ArcA380):" "$REF2" "$DIS2" 1920 1080 8 --no_cuda
echo ""

echo "============================================"
echo "  FPS Benchmark - All Backends (48f, 10-bit)"
echo "============================================"
echo ""

for res in 576x324 1280x720 1920x1080 3840x2160; do
    echo "--- $res ---"
    bench "CPU:"            "$res" --no_sycl --no_cuda
    bench "CUDA (RTX4090):" "$res" --no_sycl
    bench "SYCL (ArcA380):" "$res" --no_cuda
    echo ""
done

echo "=== All done ==="
