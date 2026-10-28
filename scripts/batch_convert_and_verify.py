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
test_videos = ["3.mp4", "5.mp4", "6.mp4", "7.mp4", "8.mp4", "15.mp4"]

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
        
    print(f"\n==========================================")
    print(f"Processing: {vname} -> {out_path.name}")
    print(f"==========================================")
    
    t0 = time.time()
    # Profile video
    prof = profiler.profile(str(in_path))
    print(f"Profile: mark_found={prof.watermark_found}, region={prof.region_str}, conf={prof.confidence}")
    
    # Process with python lossless reverse-alpha engine
    res = manager.python_engine.process_video(
        input_path=str(in_path),
        output_path=str(out_path),
        region=prof.region_str,
        log_callback=lambda msg: print(f"  [Log] {msg}")
    )
    t_elapsed = time.time() - t0
    print(f"Finished {out_path.name} in {t_elapsed:.1f}s: {res.get('status')}")
    
    # ----------------------------------------------------
    # Quality Verification directly on output video frames
    # ----------------------------------------------------
    cap_c = cv2.VideoCapture(str(out_path))
    cap_o = cv2.VideoCapture(str(in_path))
    
    total = int(cap_c.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Verifying {out_path.name} ({total} frames)...")
    
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
            print(f"  Saved verification crop: {crop_save_path.name}")
            
    cap_o.release()
    cap_c.release()
    
    avg_orig = np.mean(corrs_orig) if corrs_orig else 0.0
    avg_clean = np.mean(corrs_clean) if corrs_clean else 0.0
    print(f"  Quality metrics for {vname}:")
    print(f"    Original template correlation: {avg_orig:.3f}")
    print(f"    Cleaned template correlation:  {avg_clean:.3f} (Lower/negative is better; <0.10 means logo removed)")
    print(f"    Status: {'EXCELLENT (LOGO REMOVED)' if avg_clean < 0.15 else 'CHECK NEEDED'}")

print("\nAll videos processed and verified!")
