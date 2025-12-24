import cv2
import mediapipe as mp
import numpy as np
import math
from tkinter import filedialog
import tkinter as tk

# -----------------------------
# Helper functions
# -----------------------------
def vector_angle(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return math.degrees(np.clip(np.dot(v1, v2), -1.0, 1.0))

def calculate_head_rotation(landmarks):
    """Calculate head pitch, yaw, roll from pose landmarks"""
    nose = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    left_ear = np.array([landmarks[7].x, landmarks[7].y, landmarks[7].z])
    right_ear = np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])
    
    # Head tilt (roll) - based on ear positions
    ear_vec = right_ear - left_ear
    roll = math.atan2(ear_vec[1], ear_vec[0]) * 180 / math.pi
    
    # Head pitch and yaw approximations
    pitch = (nose[1] - (left_ear[1] + right_ear[1])/2) * 90
    yaw = nose[0] * 45
    
    return pitch, yaw, roll

# -----------------------------
# Enhanced BVH header with hands
# -----------------------------
BVH_HEADER = """HIERARCHY
ROOT Hips
{
    OFFSET 0.0 0.0 0.0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
        OFFSET 0.0 10.0 0.0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Neck
        {
            OFFSET 0.0 10.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Head
            {
                OFFSET 0.0 5.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                    OFFSET 0.0 5.0 0.0
                }
            }
        }
        JOINT LeftShoulder
        {
            OFFSET -5.0 8.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftElbow
            {
                OFFSET -5.0 0.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT LeftWrist
                {
                    OFFSET -5.0 0.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT LeftThumb
                    {
                        OFFSET -1.0 0.0 0.5
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {
                            OFFSET -1.0 0.0 0.0
                        }
                    }
                    JOINT LeftIndex
                    {
                        OFFSET -1.5 0.0 0.3
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {
                            OFFSET -1.5 0.0 0.0
                        }
                    }
                    JOINT LeftMiddle
                    {
                        OFFSET -1.5 0.0 0.0
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {
                            OFFSET -1.5 0.0 0.0
                        }
                    }
                    JOINT LeftRing
                    {
                        OFFSET -1.5 0.0 -0.3
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {
                            OFFSET -1.5 0.0 0.0
                        }
                    }
                    JOINT LeftPinky
                    {
                        OFFSET -1.5 0.0 -0.5
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {
                            OFFSET -1.5 0.0 0.0
                        }
                    }
                }
            }
        }
        JOINT RightShoulder
        {
            OFFSET 5.0 8.0 0.0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightElbow
            {
                OFFSET 5.0 0.0 0.0
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT RightWrist
                {
                    OFFSET 5.0 0.0 0.0
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT RightThumb
                    {
                        OFFSET 1.0 0.0 0.5
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {
                            OFFSET 1.0 0.0 0.0
                        }
                    }
                    JOINT RightIndex
                    {
                        OFFSET 1.5 0.0 0.3
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {
                            OFFSET 1.5 0.0 0.0
                        }
                    }
                    JOINT RightMiddle
                    {
                        OFFSET 1.5 0.0 0.0
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {
                            OFFSET 1.5 0.0 0.0
                        }
                    }
                    JOINT RightRing
                    {
                        OFFSET 1.5 0.0 -0.3
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {
                            OFFSET 1.5 0.0 0.0
                        }
                    }
                    JOINT RightPinky
                    {
                        OFFSET 1.5 0.0 -0.5
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        End Site
                        {
                            OFFSET 1.5 0.0 0.0
                        }
                    }
                }
            }
        }
    }
}
MOTION
"""
# -----------------------------
# Configuration - User Selection
# -----------------------------
def get_video_source():
    """Show prompt to choose between webcam or video file"""
    print("\n" + "="*50)
    print("   MOTION CAPTURE TO BVH")
    print("="*50)
    print("\nSelect video source:")
    print("  [W] - Use Webcam")
    print("  [F] - Select video File")
    print("="*50)
    while True:
        choice = input("\nEnter your choice (W/F): ").strip().upper()
        if choice == 'W':
            print("\n✓ Using webcam...")
            return None, True
        elif choice == 'F':
            print("\n✓ Opening file browser...")
            root = tk.Tk()
            root.withdraw() 
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                    ("All files", "*.*")
                ]
            )
            root.destroy()
            
            if file_path:
                print(f"✓ Selected: {file_path}")
                return file_path, False
            else:
                print("\nNo file selected. Please try again.")
        else:
            print("Invalid choice. Please enter 'W' or 'F'.")

# Get user's choice
INPUT_VIDEO, USE_WEBCAM = get_video_source()

OUTPUT_VIDEO = "output_processed.mp4"
OUTPUT_BVH = "output.bvh"

# -----------------------------
# MediaPipe setup - Pose + Hands
# -----------------------------
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open video source
if USE_WEBCAM:
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(INPUT_VIDEO)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0:
    fps = 30  # Default to 30 FPS if unable to read
    print("Warning: Could not read FPS from video, defaulting to 30 FPS")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Verify video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video source: {INPUT_VIDEO if not USE_WEBCAM else 'Webcam'}")
    exit(1)

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

motion_data = []
frame_count = 0

print(f"\nProcessing video... (Press 'q' to stop early)")
print(f"Input: {'Webcam' if USE_WEBCAM else INPUT_VIDEO}")
print(f"Output video: {OUTPUT_VIDEO}")
print(f"Output BVH: {OUTPUT_BVH}")
print("\nTracking: Body pose + Head rotation + Hand fingers (all 10 fingers)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process pose and hands
    pose_result = pose.process(rgb)
    hands_result = hands.process(rgb)

    # Create a copy of the frame for drawing
    output_frame = frame.copy()

    # Initialize frame data with zeros
    frame_data = [0] * 72  # Root(6) + Spine(3) + Neck(3) + Head(3) + Arms+Hands(57)

    if pose_result.pose_landmarks:
        # Draw pose skeleton (green)
        mp_drawing.draw_landmarks(
            output_frame,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0),
                thickness=2,
                circle_radius=3
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0),
                thickness=2
            )
        )

        lm = pose_result.pose_landmarks.landmark

        # Calculate head rotation
        head_pitch, head_yaw, head_roll = calculate_head_rotation(lm)

        # Torso rotation
        left_shoulder = np.array([lm[11].x, lm[11].y, lm[11].z])
        right_shoulder = np.array([lm[12].x, lm[12].y, lm[12].z])
        shoulder_vec = right_shoulder - left_shoulder
        spine_rotation = vector_angle(shoulder_vec, np.array([1, 0, 0]))

        # Arms
        left_elbow = np.array([lm[13].x, lm[13].y, lm[13].z])
        right_elbow = np.array([lm[14].x, lm[14].y, lm[14].z])
        left_wrist = np.array([lm[15].x, lm[15].y, lm[15].z])
        right_wrist = np.array([lm[16].x, lm[16].y, lm[16].z])

        left_shoulder_angle = vector_angle(left_elbow - left_shoulder, np.array([0, -1, 0]))
        right_shoulder_angle = vector_angle(right_elbow - right_shoulder, np.array([0, -1, 0]))
        left_elbow_angle = vector_angle(left_wrist - left_elbow, left_elbow - left_shoulder)
        right_elbow_angle = vector_angle(right_wrist - right_elbow, right_elbow - right_shoulder)

        # Fill frame data
        frame_data[0:6] = [0, 0, 0, 0, spine_rotation, 0]  # Root
        frame_data[6:9] = [0, spine_rotation, 0]  # Spine
        frame_data[9:12] = [0, 0, 0]  # Neck
        frame_data[12:15] = [head_roll, head_pitch, head_yaw]  # Head with rotation
        frame_data[15:18] = [0, left_shoulder_angle, 0]  # Left shoulder
        frame_data[18:21] = [0, left_elbow_angle, 0]  # Left elbow
        frame_data[21:24] = [0, 0, 0]  # Left wrist
        frame_data[39:42] = [0, right_shoulder_angle, 0]  # Right shoulder
        frame_data[42:45] = [0, right_elbow_angle, 0]  # Right elbow
        frame_data[45:48] = [0, 0, 0]  # Right wrist

    # Process hand landmarks
    if hands_result.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
            # Draw hand skeleton (cyan for better visibility)
            mp_drawing.draw_landmarks(
                output_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 255, 0),  # Cyan
                    thickness=2,
                    circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(255, 255, 0),
                    thickness=2
                )
            )

            # Determine left or right hand
            handedness = hands_result.multi_handedness[hand_idx].classification[0].label
            
            # Calculate finger rotations
            h = hand_landmarks.landmark
            
            # Thumb
            thumb_angle = vector_angle(
                np.array([h[4].x - h[3].x, h[4].y - h[3].y, h[4].z - h[3].z]),
                np.array([1, 0, 0])
            )
            
            # Index
            index_angle = vector_angle(
                np.array([h[8].x - h[6].x, h[8].y - h[6].y, h[8].z - h[6].z]),
                np.array([0, -1, 0])
            )
            
            # Middle
            middle_angle = vector_angle(
                np.array([h[12].x - h[10].x, h[12].y - h[10].y, h[12].z - h[10].z]),
                np.array([0, -1, 0])
            )
            
            # Ring
            ring_angle = vector_angle(
                np.array([h[16].x - h[14].x, h[16].y - h[14].y, h[16].z - h[14].z]),
                np.array([0, -1, 0])
            )
            
            # Pinky
            pinky_angle = vector_angle(
                np.array([h[20].x - h[18].x, h[20].y - h[18].y, h[20].z - h[18].z]),
                np.array([0, -1, 0])
            )

            # Store finger data
            if handedness == "Left":
                frame_data[24:27] = [0, thumb_angle, 0]
                frame_data[27:30] = [0, index_angle, 0]
                frame_data[30:33] = [0, middle_angle, 0]
                frame_data[33:36] = [0, ring_angle, 0]
                frame_data[36:39] = [0, pinky_angle, 0]
            else:  # Right hand
                frame_data[48:51] = [0, thumb_angle, 0]
                frame_data[51:54] = [0, index_angle, 0]
                frame_data[54:57] = [0, middle_angle, 0]
                frame_data[57:60] = [0, ring_angle, 0]
                frame_data[60:63] = [0, pinky_angle, 0]

    motion_data.append(frame_data)

    # Write the frame with skeleton overlay
    out.write(output_frame)

    # Display progress
    cv2.imshow("Processing - Body (Green) + Hands (Cyan)", output_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Stopped by user")
        break

    # Print progress every 30 frames
    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✓ Processing complete!")
print(f"Total frames processed: {frame_count}")

# Write BVH file
with open(OUTPUT_BVH, "w") as f:
    f.write(BVH_HEADER)
    f.write(f"Frames: {len(motion_data)}\n")
    f.write(f"Frame Time: {1.0/fps:.7f}\n")
    for frame in motion_data:
        f.write(" ".join(map(str, frame)) + "\n")

print(f"✓ Saved animation to {OUTPUT_BVH}")
print(f"✓ Saved processed video with skeleton to {OUTPUT_VIDEO}")