import cv2 as cv
import numpy as np

# Configuration parameters
whT = 320
confThreshold = 0.5
nmsThreshold = 0.2

# Load class names from coco.names
classesFile = "coco.names"
classNames = []
try:
    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
except FileNotFoundError:
    print(f"Error: {classesFile} not found. Ensure the file is in the same directory.")
    exit()

# Model files for YOLOv3-tiny
modelConfiguration = "yolov3-tiny.cfg"
modelWeights = "yolov3-tiny.weights"

# Load the YOLO model
try:
    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
except cv.error as e:
    print(f"Error loading model: {e}")
    exit()

def findObjects(outputs, img):
    """Process the model outputs and draw bounding boxes on the image."""
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    # Parse detections
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    # Apply Non-Maximum Suppression
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    if len(indices) > 0:  # Ensure indices are not empty
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                        (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

def main():
    """Main function to capture video and process frames."""
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        exit()

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Unable to read frame from camera.")
            break

        # Prepare the image for YOLO
        blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)

        # Get layer names and output layer names
        layersNames = net.getLayerNames()
        try:
            unconnectedLayers = net.getUnconnectedOutLayers()
            unconnectedLayers = np.array(unconnectedLayers).flatten()  # Ensure it's an array
            outputNames = [layersNames[i - 1] for i in unconnectedLayers]
        except Exception as e:
            print(f"Error processing layer names: {e}")
            break

        # Forward pass to get outputs
        outputs = net.forward(outputNames)

        # Find and draw objects
        findObjects(outputs, img)

        # Display the frame
        cv.imshow('YOLOv3-Tiny Object Detection', img)

        # Exit on pressing 'q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
