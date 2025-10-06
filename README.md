# MultiObjectTracker  
**A Lightweight Multi-Object Tracking (MOT) Demo Based on DeepSORT**

**Keywords**: Multi-Object Tracking, YOLOv12n, MNN Inference, ONNX Runtime, ReID, Kalman Filter

---

## 📌 Overview  
This project implements a simple yet effective **multi-object tracking (MOT)** system using the **DeepSORT** algorithm. It combines modern object detection with appearance-based re-identification (ReID) to achieve robust tracking across video frames.

Key components:
- **Object Detection**: Uses **YOLOv12n** (a lightweight YOLO variant) via **ONNX Runtime** for fast and efficient inference.
- **Appearance Embedding**: Employs **OSNet_x1_0** as the ReID model, accelerated by the **MNN inference engine** for real-time feature extraction.
- **Tracking Logic**: Integrates **Kalman Filter** for motion prediction and **DeepSORT** for data association using both motion and appearance cues.

---

## 🚀 Quick Start  

1. **Build the project**:
   ```bash
   ./build.sh
   ```
   Upon successful compilation, a `build/` directory will be created containing all executables.

2. **Run the tracker demo**:
   ```bash
   ./build/test_tracker
   ```
   The executable names in `build/` correspond to the source files in the `test/` directory.

> 💡 Ensure that required dependencies (MNN, ONNX Runtime, OpenCV, etc.) are properly installed before building.

---

## 🙏 Acknowledgements  
This work builds upon and draws inspiration from the following excellent open-source projects:
- [UCMCTrack](https://github.com/corfyi/UCMCTrack)  
- [shaoshengsong/DeepSORT](https://github.com/shaoshengsong/DeepSORT)  
- [nwojke/deep_sort](https://github.com/nwojke/deep_sort)  

We sincerely thank the authors for their contributions to the MOT community.

---

## 🛠️ Roadmap & Future Work  
As of the **2025.10.06** version, several aspects are under active development:

1. **Improve Tracking Accuracy**  
   - Refine association thresholds  
   - Enhance handling of occlusions and ID switches

2. **Optimize Runtime Performance**  
   - Reduce end-to-end latency  
   - Profile and accelerate ReID and detection pipelines

3. **Model Upgrades**  
   - Evaluate more accurate/efficient object detectors (e.g., YOLOv11, YOLO-NAS)  
   - Experiment with lightweight ReID models (e.g., OSNet-AIN, LReID)

Contributions and suggestions are welcome!

---
