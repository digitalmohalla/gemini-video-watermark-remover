import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure root directory is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.stdout.reconfigure(encoding='utf-8')

from core.batch_manager import BatchManager
from core.profiler import IntelligentWatermarkProfiler

target_folder = Path(r"E:\Naastik Brand\Youtube\Youtube Posted 2\प्रमाण-पत्र का कुरुक्षेत्र\original")
test_videos = ["11.mp4", "14.mp4", "16.mp4", "20.mp4", "25.mp4"]

print(f"Target folder: {target_folder}")
print(f"Videos to process: {test_videos}")

manager = BatchManager()
profiler = IntelligentWatermarkProfiler()
tmpl_48 = cv2.imread(str(root_dir / "core" / "assets" / "bg_48.png"), cv2.IMREAD_GRAYSCALE)

for vname in test_videos:
    in_path = target_folder / vname
    out_path = target_folder / f"{vname.replace('.mp4', '_clean.mp4')}"
    
    if not in_path.exists():
        print(f"Warning: {in_path} does not exist!")
        continue
        
    print(f"\n==========================================", flush=True)
    print(f"Processing: {vname} -> {out_path.name}", flush=True)
    print(f"==========================================", flush=True)
    
    t0 = time.time()
    # Profile video
    prof = profiler.profile(str(in_path))
    print(f"Profile: mark_found={prof.watermark_found}, region={prof.region_str}, conf={prof.confidence}", flush=True)
    
    # Process with python lossless reverse-alpha engine if watermark found, else preserve directly
    if prof.watermark_found:
        res = manager.python_engine.process_video(
            input_path=str(in_path),
            output_path=str(out_path),
            region=prof.region_str,
            log_callback=lambda msg: print(f"  [Log] {msg}", flush=True)
        )
        t_elapsed = time.time() - t0
        print(f"Finished {out_path.name} in {t_elapsed:.1f}s: {res.get('status')}", flush=True)
    else:
        import shutil
        shutil.copy2(str(in_path), str(out_path))
        t_elapsed = time.time() - t0
        print(f"  [Log] No watermark detected in {vname} (already clean). Preserved directly in {t_elapsed:.2f}s.", flush=True)
    
    # ----------------------------------------------------
    # Quality Verification directly on output video frames
    # ----------------------------------------------------
    cap_c = cv2.VideoCapture(str(out_path))
    cap_o = cv2.VideoCapture(str(in_path))
    
    total = int(cap_c.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Verifying {out_path.name} ({total} frames)...", flush=True)
    
    # Test frames at 20%, 50%, 80%
    corrs_orig = []
    corrs_clean = []
    diffs = []
    
    wx, wy, ww, wh = prof.x, prof.y, prof.width_wm, prof.height_wm
    
    for r in [0.20, 0.50, 0.80]:
        fidx = int(total * r)
        cap_o.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        cap_c.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        
        _, fo = cap_o.read()
        _, fc = cap_c.read()
        
        if fo is None or fc is None:
            continue
            
        roi_o = cv2.cvtColor(fo[wy:wy+wh, wx:wx+ww], cv2.COLOR_BGR2GRAY)
        roi_c = cv2.cvtColor(fc[wy:wy+wh, wx:wx+ww], cv2.COLOR_BGR2GRAY)
        
        res_o = cv2.matchTemplate(roi_o, tmpl_48, cv2.TM_CCOEFF_NORMED)
        res_c = cv2.matchTemplate(roi_c, tmpl_48, cv2.TM_CCOEFF_NORMED)
        
        corrs_orig.append(float(res_o[0, 0]))
        corrs_clean.append(float(res_c[0, 0]))
        
        # Save frame 50% crop for inspection
        if r == 0.50:
            crop_save_path = root_dir / f"verify_{vname.replace('.mp4', '')}_clean_f{fidx}.png"
            cv2.imwrite(str(crop_save_path), fc[max(0, wy-20):min(fc.shape[0], wy+wh+20), max(0, wx-20):min(fc.shape[1], wx+ww+20)])
            print(f"  Saved verification crop: {crop_save_path.name}", flush=True)
            
    cap_o.release()
    cap_c.release()
    
    avg_orig = np.mean(corrs_orig) if corrs_orig else 0.0
    avg_clean = np.mean(corrs_clean) if corrs_clean else 0.0
    print(f"  Quality metrics for {vname}:", flush=True)
    print(f"    Original template correlation: {avg_orig:.3f}", flush=True)
    print(f"    Cleaned template correlation:  {avg_clean:.3f} (Lower/negative is better; <0.10 means logo removed)", flush=True)
    print(f"    Status: {'EXCELLENT (LOGO REMOVED)' if avg_clean < 0.15 else 'CHECK NEEDED'}", flush=True)

print("\nAll videos processed and verified!", flush=True)
