from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from google.cloud import vision
import json
import cv2 as cv
from PIL import Image

import os

@csrf_exempt
def ocr(request):
    if request.method == 'POST':
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "D:\ocr_project\ocr-project-441501-96dba7ee3fe8.json"
        data = json.loads(request.body)
        image_path = data.get("imagePath")

        print(image_path)

        client = vision.ImageAnnotatorClient()
        img = cv.imread(image_path)
        rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        _, encoded_image = cv.imencode('.PNG', rgb_img)
        content = encoded_image.tobytes()

        input_img = vision.Image(content=content)
        response = client.text_detection(image=input_img)

        texts = response.text_annotations
        if texts:
            detected_text = texts[0].description
            return JsonResponse({'result': detected_text}, status=200)

        return JsonResponse({'error': 'No text detected'}, status=400)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
