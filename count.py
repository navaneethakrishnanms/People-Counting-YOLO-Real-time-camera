import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import cv2
import collections
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


MODEL_PATH = "yolov8s.pt"
RTSP_URL = "rtsp://gopal:bitsathy@123@10.10.131.30:554/live.sdp"
CONFIDENCE_THRESHOLD = 0.5
GPU_DEVICE = 0 


def get_orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def on_segment(p, q, r):
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def do_lines_intersect(p1, q1, p2, q2):
    o1, o2 = get_orientation(p1, q1, p2), get_orientation(p1, q1, q2)
    o3, o4 = get_orientation(p2, q2, p1), get_orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4: return True
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True
    return False


print("Loading YOLO model...")
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()
tracker = DeepSort(max_age=30, n_init=3)


print(f"Connecting to RTSP stream to capture a setup frame...")
cap_setup = cv2.VideoCapture(RTSP_URL)
if not cap_setup.isOpened():
    print("Error: Cannot open RTSP stream for setup.")
    exit()
ret, first_frame = cap_setup.read()
if not ret:
    print("Failed to read the first frame from the stream for setup.")
    cap_setup.release()
    exit()
cap_setup.release() 
print("Setup frame captured. Please draw the line.")



line_points = []
def select_line_event(event, x, y, flags, param):
    global line_points, first_frame
    if event == cv2.EVENT_LBUTTONDOWN and len(line_points) < 2:
        line_points.append((x, y))
        print(f"Point selected: ({x}, {y})")
        cv2.circle(first_frame, (x, y), 5, (0, 0, 255), -1)
        if len(line_points) == 2:
            cv2.line(first_frame, line_points[0], line_points[1], (0, 0, 255), 2)
        cv2.imshow("Select Counting Line", first_frame)

print("\nPlease select two points on the window to draw the counting line.")
print("Press any key after selecting two points to continue.")
cv2.imshow("Select Counting Line", first_frame)
cv2.setMouseCallback("Select Counting Line", select_line_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(line_points) != 2:
    print("Error: You must select exactly two points. Exiting.")
    exit()
line_start, line_end = line_points


print(f"\nLine selected. Re-connecting to RTSP stream for live processing...")
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    print("Error: Failed to re-open RTSP stream for processing.")
    exit()


counter = 0
crossed_ids = set()
trajectories = collections.defaultdict(lambda: collections.deque(maxlen=30))



print("Starting video processing...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream ended or failed to grab frame.")
        break

    results = model(frame, device=GPU_DEVICE, verbose=False)[0]
    detections = []
    try: person_class_id = list(model.names.keys())[list(model.names.values()).index("person")]
    except ValueError: person_class_id = 0

    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) == person_class_id and conf > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf.item(), "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed(): continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        trajectories[track_id].append((center_x, center_y))

        if len(trajectories[track_id]) >= 2:
            prev_point = trajectories[track_id][-2]
            current_point = trajectories[track_id][-1]
            if track_id not in crossed_ids and do_lines_intersect(prev_point, current_point, line_start, line_end):
                crossed_ids.add(track_id)
                counter += 1

       
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for point in trajectories[track_id]: cv2.circle(frame, point, 3, (0, 255, 255), -1)

    cv2.line(frame, line_start, line_end, (0, 0, 255), 2)
    cv2.putText(frame, f"People Crossed: {counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.imshow("People Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break


cap.release()
cv2.destroyAllWindows()