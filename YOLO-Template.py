import torch
import cv2
import time

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)

#Change yolov5 to any model depending on processing power
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if ret is None:
        break

    start_time = time.time()

    result = model(frame)
    df = result.pandas().xyxy[0]
    # print(df)
    for ind in df.index:
        x1, y1 = int(df['xmin'][ind]), int(df['ymin'][ind])
        x2, y2 = int(df['xmax'][ind]), int(df['ymax'][ind])
        label = df['name'][ind]
        conf = df['confidence'][ind]
        text = label + ' ' + str(conf.round(decimals= 2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

    cv2.imshow('Video', frame)

    end_time = time.time()

    process_time = end_time - start_time
    print(f"Frame processing time: {process_time:.4f} seconds")

webcam.release()
cv2.destroyAllWindows()
