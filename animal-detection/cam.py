from dotenv import load_dotenv
import urllib.request
import numpy as np
import cv2
from inference import get_model
import time

load_dotenv() 

model = get_model(model_id="my-first-project-qvkry/2")
url = 'http://192.168.1.187/cam-hi.jpg'

confThreshold = 0.7

while True:
    # Fetch snapshot
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, -1)

    # Inference
    results = model.infer(im)

    for response in results:
        for prediction in response.predictions:
            x = int(prediction.x)
            y = int(prediction.y)
            w = int(prediction.width)
            h = int(prediction.height)
            class_name = prediction.class_name
            confidence = prediction.confidence

            x1 = max(x - w // 2, 0)
            y1 = max(y - h // 2, 0)
            x2 = min(x + w // 2, im.shape[1])
            y2 = min(y + h // 2, im.shape[0])

            if confidence > confThreshold and class_name in ['squirrel', 'bat', 'butterfly']:
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(im, f"{class_name} ({confidence:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                print(f'ALERT: {class_name.upper()} detected! Confidence: {confidence:.2f}')
            else:
                cv2.putText(im, 'SAFE!', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Camera', im)

    if cv2.waitKey(1) == ord('q'):
        break

    # Optional: throttle frame rate
    time.sleep(0.1)

cv2.destroyAllWindows()
