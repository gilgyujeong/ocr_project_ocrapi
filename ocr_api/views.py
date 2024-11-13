from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from google.cloud import vision
import json
import cv2 as cv
from ultralytics import YOLO
import re

import os

@csrf_exempt
def ocr(request):
    if request.method == 'POST':
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\ocr_project\ocr-project-441501-96dba7ee3fe8.json"
        data = json.loads(request.body)
        image_path = data.get("imagePath")

        print(image_path)

        client = vision.ImageAnnotatorClient()

        model = YOLO('../models/best.pt')
        result = model.predict(image_path)

        img = cv.imread(image_path)

        x1, y1, x2, y2 = map(int, result[0].boxes.xyxy[0])
        # cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cut_img = img[y1:y2, x1:x2]

        cut_img_path = f'C:/Users/AI-00/Desktop/ocr_project/ocr_project_frontend/ocr_project/public/upload/detection_img/{str(image_path).split('/')[9]}'
        cv.imwrite(cut_img_path, cut_img)

        rgb_img = cv.cvtColor(cut_img, cv.COLOR_BGR2RGB)
        _, encoded_image = cv.imencode('.PNG', rgb_img)
        content = encoded_image.tobytes()

        input_img = vision.Image(content=content)
        response = client.text_detection(image=input_img)

        texts = response.text_annotations
        if texts:
            detected_text = texts[0].description
            numbers_only = re.findall(r'\d+', detected_text)
            return JsonResponse({'result': numbers_only[-1]}, status=200)

        return JsonResponse({'error': 'No text detected'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
