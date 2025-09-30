# 👥 People Counting with YOLOv8 + DeepSORT

🚀 A real-time **people counting system** using **YOLOv8 object detection** + **DeepSORT tracking**, with an interactive **line crossing counter**.  
Ideal for **crowd analytics, retail, smart surveillance, and traffic monitoring**.

---

## ✨ Features
- 🎯 **YOLOv8** for accurate person detection  
- 🔄 **DeepSORT** for robust multi-object tracking with unique IDs  
- 🖱️ Interactive **line drawing** (select 2 points on the video to define your counting line)  
- 🔢 **Crossing counter** to count people crossing the line without duplicates  
- 📡 **RTSP camera support** (works with IP cameras, CCTV, or live streams)  
- ⚡ GPU acceleration with CUDA (optional)  

---

## 📸 Demo
<p align="center">
  <img src="assets/demo.gif" alt="YOLOv8 DeepSORT People Counting Demo" width="700">
</p>

---

## 🛠️ Installation

### 1️⃣ Clone this repo
```bash
git clone https://github.com/your-username/people-counting-yolov8.git
cd people-counting-yolov8

2️⃣ Create a conda environment
conda create -n peoplecount python=3.9 -y
conda activate peoplecount

3️⃣ Install dependencies
pip install ultralytics opencv-python deep-sort-realtime


✅ Make sure you have PyTorch with CUDA installed if you want GPU acceleration.
Check with:

python -c "import torch; print(torch.cuda.is_available())"

🚀 Usage
Run the People Counter
python people_counter.py

Arguments inside the script:

MODEL_PATH → path to YOLOv8 model (default: yolov8s.pt)

RTSP_URL → your RTSP camera stream URL

CONFIDENCE_THRESHOLD → minimum confidence for detection (default: 0.5)

GPU_DEVICE → set 0 for GPU, or cpu

🎮 How It Works

Connects to your RTSP camera.

Captures the first frame → lets you draw a line with 2 mouse clicks.

Runs YOLOv8 → detects persons.

Tracks them with DeepSORT → assigns IDs.

Checks if a person’s path crosses the line → increments counter.

Displays real-time bounding boxes, IDs, trajectories, and total count.

📂 Project Structure
📦 people-counting-yolov8
 ┣ 📜 people_counter.py     # Main script
 ┣ 📜 requirements.txt      # Dependencies
 ┣ 📜 README.md             # This file
 ┗ 📂 assets
    ┗ 📜 demo.gif           # Demo video/image

🔮 Future Improvements

 Zone-based counting (not just lines)

 Web dashboard with live analytics

 Support for multiple cameras

 Export results (CSV / database logging)

🤝 Contributing

Pull requests are welcome!
If you’d like to improve detection, add new features, or fix bugs, feel free to fork and submit a PR.# People-Counting-YOLO-Real-time-camera
