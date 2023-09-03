from roboflow import Roboflow
import cv2
import matplotlib.pyplot as plt

rf = Roboflow(api_key="ir7oLC7KgiWuNwFLjqYn")
project = rf.workspace().project("barbells-detector")
model = project.version(10).model

# Load the video
video_path = 'cc.mp4'
cap = cv2.VideoCapture(video_path)

target_positions = []
target_confidences = []

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (width, height))

while True:

    ret, frame = cap.read()

    if not ret:
        break

    temp_file = 'temp.jpg'
    cv2.imwrite(temp_file, frame)

    # Evaluate the image using Roboflow API
    predictions = model.predict(temp_file, confidence=20, overlap=30).json()

    # Select the highest confidence as the target
    end_predictions = [p for p in predictions['predictions'] if p['class'] == 'Barbell']
    if end_predictions:
        highest_confidence_prediction = max(end_predictions, key=lambda x: x['confidence'])

        # Perform outlier detection
        if len(target_positions) == 0 or (abs(highest_confidence_prediction['x'] - target_positions[-1][0]) <= 50 and abs(highest_confidence_prediction['y'] - target_positions[-1][1]) <= 50):
            target_positions.append((int(highest_confidence_prediction['x']), int(highest_confidence_prediction['y'])))
            target_confidences.append(highest_confidence_prediction['confidence'])

        else:
            target_positions.append(target_positions[-1])
            target_confidences.append(target_confidences[-1])

    else:
        if len(target_positions) > 0:
            target_positions.append(target_positions[-1])
            target_confidences.append(target_confidences[-1])

    # Display the frame with the trajectory overlay
    for i in range(len(target_positions)):
        # if target_positions[i] is not None:
            # cv2.circle(frame, target_positions[i], radius=2, color=(0, 0, 255), thickness=-1)
        if i > 0 and target_positions[i-1] is not None and target_positions[i] is not None:
            cv2.line(frame, target_positions[i-1], target_positions[i], color=(0, 128, 255), thickness=2)

    out.write(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()

fig, ax = plt.subplots()
ax.plot([p[0] for p in target_positions if p is not None], [p[1] for p in target_positions if p is not None], marker='o')
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_title('Trajectory of Target')
plt.show()
