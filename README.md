# 📐 Data-Driven Distance Calculation and Compliance Assessment for Photovoltaic System Installations

**Author**: Raniyaharini Rajendran  
**Institution**: Freie Universität Berlin – Institute of Mathematics and Computer Science  
**Thesis Type**: MS Data Science

**Date of Submission**: 28.02.2024

> ⚠️ **Note**: This project contains confidential content developed in collaboration with Enpal Asset Management GmbH. The thesis is not for public distribution without proper authorization.

---

## 🧠 Project Summary

This project explores the application of both **traditional** and **Transformer-based deep learning models** for:

- **Object detection** of photovoltaic (PV) system components
- **Distance measurement** between key components (e.g., roof hooks)
- **Compliance assessment** against installation standards

Rather than merging model types, the study evaluates the strengths and limitations of each approach **individually**, focusing on their **accuracy, precision, and robustness** in real-world solar installation scenarios.

---

## 🎯 Objectives

- Develop a proof of concept for **automated compliance checks** in PV installations.
- Compare YOLOv6, YOLO-NAS, and DETR (Detection Transformers) for object detection tasks.
- Automate **distance calculations** between critical PV components (e.g., hooks).
- Analyze model performance for enhancing **quality control**, reducing **manual inspection** time, and ensuring **safety standards**.

---

## 🛠️ Methods & Tools

### Models Compared:
- **YOLOv6** – Fast and accurate object detection
- **YOLO-NAS** – Optimized via Neural Architecture Search
- **DETR** – Transformer-based model treating detection as set prediction

### Dataset:
- ~15,000 annotated images of rooftop PV installations from **Enpal GmbH**'s internal database.
- Labeling done using [makesense.ai](https://www.makesense.ai/)

### Frameworks & Libraries:
- **PyTorch**
- **TensorFlow**
- **Darknet**
- **LabelImg / makesense.ai** (for annotations)
- **OpenCV** (for image processing)
- **CUDA** (for GPU acceleration)

---

## 📊 Evaluation Metrics

- **Precision**
- **Recall**
- **F1-Score**
- **mAP (mean Average Precision)**
- **IoU (Intersection over Union)**

---

## 📸 Key Features

- Automated detection of hooks on slanted and flat rooftops
- Pixel-based and centimeter-level distance measurement
- Evaluation of performance degradation across varying lighting, angle, and resolution conditions
- Visual compliance check for hook placement alignment

---

## 📈 Results Overview

| Model       | mAP (%) | Best Use Case                    |
|-------------|---------|----------------------------------|
| YOLOv6      | 46.4–52 | Fast real-time detection         |
| YOLO-NAS    | 47.5–52 | Improved accuracy via NAS        |
| DETR        | ~69     | Holistic understanding of scenes |

> DETR demonstrated high accuracy in controlled environments but required more training time and resources.

---

## 🔬 Research Contribution

- First known comparison of **YOLOv6, YOLO-NAS, and DETR** for compliance verification in PV installations
- Demonstrates **Transformer effectiveness** in high-precision industrial quality control
- Proposes a **scalable automation pipeline** for image-based compliance assessment

---

## 📬 Contact

**Author**: Raniyaharini Rajendran  
**Email**: r.raniyaharini@gmail.com

**Supervisor**: Dr. rer. nat. Vitaly Belik  
**Examiner**: Prof. Dr. Katinka Wolter
