#!/bin/bash
source /opt/intel/oneapi/setvars.sh 2>/dev/null
cd /home/kilian/dev/libvmaf_vulkan

VMAF=/usr/local/bin/vmaf
MODEL=model/vmaf_v0.6.1.json
OUTDIR=testdata/bbb/results
mkdir -p "$OUTDIR"

run() {
    local name="$1" ref="$2" dis="$3" w="$4" h="$5" bd="$6" flags="$7"
    local out="$OUTDIR/${name}.json"
    echo -n "  $name ... "
    local start=$(date +%s%N)
    $VMAF --reference "$ref" --distorted "$dis" \
        --width $w --height $h --pixel_format 420 --bitdepth $bd \
        --model path=$MODEL --threads 1 \
        --output "$out" --json -q $flags 2>/dev/null
    local end=$(date +%s%N)
    local ms=$(( (end - start) / 1000000 ))
    local score=$(python3 -c "import json; print(f'{json.load(open(\"$out\"))[\"pooled_metrics\"][\"vmaf\"][\"mean\"]:.6f}')")
    echo "${score}  (${ms}ms)"
}

compare() {
    local tag="$1"
    python3 - "$tag" "$OUTDIR" <<'PYEOF'
import json, sys
tag = sys.argv[1]
outdir = sys.argv[2]
backends = [
    (f"{tag}_cpu", "CPU"),
    (f"{tag}_cuda", "CUDA RTX4090"),
    (f"{tag}_sycl", "SYCL Arc A380"),
]
cpu_scores = None
for key, name in backends:
    try:
        with open(f"{outdir}/{key}.json") as f:
            d = json.load(f)
        scores = [fr["metrics"]["vmaf"] for fr in d["frames"]]
        mean = d["pooled_metrics"]["vmaf"]["mean"]
        if cpu_scores is None:
            cpu_scores = scores
            print(f"  {name:20s}: {mean:.6f} (ref, {len(scores)} frames)")
        else:
            diffs = [abs(c-g) for c,g in zip(cpu_scores, scores)]
            mx = max(diffs)
            avg = sum(diffs)/len(diffs)
            bad = [(i,d) for i,d in enumerate(diffs) if d > 0.01]
            st = "PASS" if mx < 0.01 else ("WARN" if mx < 0.1 else "FAIL")
            print(f"  {name:20s}: {mean:.6f}  {st} max_diff={mx:.8f} avg_diff={avg:.8f}")
            if bad:
                for i,d in bad[:5]:
                    print(f"      frame {i}: cpu={cpu_scores[i]:.6f} gpu={scores[i]:.6f} diff={d:.6f}")
    except Exception as e:
        print(f"  {name:20s}: ERROR {e}")
PYEOF
}

echo "========================================="
echo "Test 1: Official 576x324 (48 frames, 8-bit)"
echo "========================================="
REF=python/test/resource/yuv/src01_hrc00_576x324.yuv
DIS=python/test/resource/yuv/src01_hrc01_576x324.yuv
run "t1_cpu"    "$REF" "$DIS" 576 324 8 "--no_cuda --no_sycl"
run "t1_cuda"   "$REF" "$DIS" 576 324 8 "--no_sycl"
run "t1_sycl"   "$REF" "$DIS" 576 324 8 "--no_cuda"
echo "Comparison:"
compare "t1"

echo ""
echo "========================================="
echo "Test 2: Official 1080p (5 frames, 8-bit)"
echo "========================================="
REF=python/test/resource/yuv/src01_hrc00_1920x1080_5frames.yuv
DIS=python/test/resource/yuv/src01_hrc01_1920x1080_5frames.yuv
run "t2_cpu"    "$REF" "$DIS" 1920 1080 8 "--no_cuda --no_sycl"
run "t2_cuda"   "$REF" "$DIS" 1920 1080 8 "--no_sycl"
run "t2_sycl"   "$REF" "$DIS" 1920 1080 8 "--no_cuda"
echo "Comparison:"
compare "t2"

echo ""
echo "========================================="
echo "Test 3: BBB 4K (200 frames, 8-bit)"
echo "========================================="
REF=testdata/bbb/ref_3840x2160_200f.yuv
DIS=testdata/bbb/dis_3840x2160_200f.yuv
run "t3_cpu"    "$REF" "$DIS" 3840 2160 8 "--no_cuda --no_sycl"
run "t3_cuda"   "$REF" "$DIS" 3840 2160 8 "--no_sycl"
run "t3_sycl"   "$REF" "$DIS" 3840 2160 8 "--no_cuda"
echo "Comparison:"
compare "t3"

echo ""
echo "========================================="
echo "DONE"
echo "========================================="
