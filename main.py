import os
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.saved_model.load(os.path.join('saved_model'))

# Load the label map for the classes that the model can detect
label_map = {'1': 'normal', '2': 'Osteoarthritis'} 

# Load the image to be detected
image = cv2.imread('path/to/image')

# Convert the image to a format that the model can use
input_tensor = tf.convert_to_tensor(image)
input_tensor = input_tensor[tf.newaxis, ...]

# Use the model to detect the image
detections = model(input_tensor)

# Process the detections
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
detections['num_detections'] = num_detections

# Filter out low confidence detections
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
scores = detections['detection_scores']
detection_threshold = 0.3
detections['detection_boxes'] = detections['detection_boxes'][scores > detection_threshold]
detections['detection_classes'] = detections['detection_classes'][scores > detection_threshold]
detections['detection_scores'] = detections['detection_scores'][scores > detection_threshold]

# Drawing the detections on the image
for i in range(len(detections['detection_boxes'])):
    box = detections['detection_boxes'][i]
    class_id = detections['detection_classes'][i]
    class_name = label_map[str(class_id)]
    score = detections['detection_scores'][i]
    xmin, ymin, xmax, ymax = box
    xmin = int(xmin * image.shape[1])
    xmax = int(xmax * image.shape[1])
    ymin = int(ymin * image.shape[0])
    ymax = int(ymax * image.shape[0])
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(image, '{}: {:.2f}'.format(class_name, score), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#Selecting the one with highest score
highest_score_idx = np.argmax(detections['detection_scores'])

# Get the class ID and class name for the detection with the highest score
class_id = detections['detection_classes'][highest_score_idx]
class_name = label_map[str(class_id)]

# Print the class with the highest probability
print('Detected type with highest probability: {}'.format(class_name))

# Display the image with the detections
cv2.imshow('Image with Detections', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
