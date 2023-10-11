import cv2
from RASC import RASC_Detector, ModelType

model_path = "***"
model_type = ModelType.RASC
use_gpu = True

# Initialize video
cap = cv2.VideoCapture("***")

# Initialize lane detection model
lane_detector = RASC_Detector(model_path, model_type, use_gpu)

cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	

while cap.isOpened():
	try:
		# Read frame from the video
		ret, frame = cap.read()
	except:
		continue

	if ret:	

		# Detect the lanes
		output_img = lane_detector.detect_lanes(frame)

		cv2.imshow("Detected lanes", output_img)

	else:
		break

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()