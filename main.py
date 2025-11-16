from wfcr import WFCR
from gui import Window
from PIL import Image
from qreader import QReader
import cv2
import requests
import json


def get_prediction(input_image):

    img, img_bbox, img_mask = reader.detect(input_image, threshold=0.8) #отправка изображений в детектор (можно изменить точность детекции параметром threshold(по умолчанию 80%))

    result, indx = reader.recognize_values(img_bbox)
    qrCode = qreader.detect_and_decode(image=cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2RGB))
    
    if indx > 1:
        img_bbox = img_bbox.rotate(180)

    window.result_window(img_bbox, result, qrCode, win_width=500) #выводим результат в отдельном окне (можно изменить размер фотографии параметром win_width)
   
    #msg = result
    #url = "http://127.0.0.1:8000/api/data"
    #data = {"meter": msg}
    #response = requests.post(url, data=data)
    #print(response.text) 
def get_value(input_image):

    img, img_bbox, img_mask = reader.detect(input_image, threshold=0.8) #отправка изображений в детектор (можно изменить точность детекции параметром threshold(по умолчанию 80%))
    qrCode = qreader.detect_and_decode(image=cv2.cvtColor(cv2.imread(input_image), cv2.COLOR_BGR2RGB))
    result = reader.recognize_values(img_bbox)[0]

    return result,qrCode




if __name__ == "__main__":
    reader = WFCR()
    qreader = QReader()
    window = Window(get_prediction)
    window.mainloop()
    result,qrCode = get_value('E:\WFCR_user\photo\IMG_20231023_110108.jpg')
    print(f"RESULT = {type(result)}")
    url = "http://127.0.0.1:8000/api/data/"
    data = {
    "meter": f"{result}",
    "qr": f"{qrCode}"
    }
    response = requests.post(url, data=data)
    print(response.json)