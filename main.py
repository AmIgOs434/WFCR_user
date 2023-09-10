from PIL import Image, ImageTk
from wfcr import WFCR
from gui import Window

def get_prediction(path):
    image = Image.open(path) #открываем изображение

    img, val_b, val_a = reader.detect(image, threshold=0.8) #отправка изображений в детектор (можно изменить точность детекции параметром threshold(по умолчанию 80%))
    result = ""

    if type(val_b).__name__ == 'ndarray': #если детектор обнаружил значения перед запятой
        pred_val_b = reader.recognize_values(val_b) #отправляем их в распознаватель
        if len(pred_val_b) > 5:
            pred_val_b = pred_val_b[:5] #отбрасываем лишние значения с конца (по умолчанию не больше 5)
        result += pred_val_b
    
    if type(val_a).__name__ == 'ndarray':
        pred_val_a = reader.recognize_values(val_a) #если детектор обнаружил значения после запятой
        if len(pred_val_a) > 3:
                pred_val_a = pred_val_a[-3:] #отбрасываем лишние значения с начала (по умолчанию не больше 3)
        result += f",{pred_val_a}"
    
    window.result_window(img, result, win_width=300) #выводим результат в отдельном окне (можно изменить размер фотографии параметром win_width)
    

if __name__ == "__main__":
    reader = WFCR()
    window = Window(get_prediction)
    window.mainloop()
    
    
    
    
    
