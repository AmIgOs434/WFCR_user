import torch
import numpy as np
import cv2
import yaml
import copy
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from recognition.utils import AttrDict, AttnLabelConverter
from recognition.model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WFCR():
    """
    Данный класс при инициализации подгружает модель детектора и распознавателя,
    которые хранятся в папках на локальном компьютере. Путь к ним передается в качестве аргумента 
    по умолчанию:
        путь к детектору - 'detr-resnet-50_finetuned_WFCR3'
        путь к распознавателю - recognition\wfcr_model.pth)
        путь к настройкам модеои - 'recognition\wfcr_model_config.yaml'

    Методы класса:
    detect: производит детецию областей, содержащих показания счетчика потребления воды. 
            Отдельно определяет области с показаниями до запятой и после запятой. 
            Возможно задать точность ниже которой область будет считаться 
            не распознанной (по умолчанию - 80%).

            Возвращает кортеж из трех изображений:
                - общее фото с рамками вокруг найденных обдастей, содержащих показания (или исходное)
                - обрезанное фото с показаниями до запятой (или None)
                - обрезанное фото с показаниями после запятой (или None)

    recognize_values: получает на вход изображение области, содержащей показания,
        предварительно обрабатывает изображение путем наложение фильтров, производит
        распознание цифровой информации. 
        Возвращает показания области в формате строки (например: '00035', '256' ...) 
    """

    def __init__(
            self, 
            det_path = 'detr-resnet-50_finetuned_WFCR3', 
            rec_path = 'recognition\wfcr_model.pth',
            rec_conf_path = 'recognition\wfcr_model_config.yaml'
        ):
        #загрузка модели детектора
        self.det_img_processor = AutoImageProcessor.from_pretrained(det_path)
        self.det_model = AutoModelForObjectDetection.from_pretrained(det_path)

        #загрузка модели распознавателя
        self.opt = self.get_config(rec_conf_path)
        self.state_dict = self.prepare_dict(torch.load(rec_path, map_location=device))
        self.rec_model = Model(self.opt)
        self.rec_model.load_state_dict(self.state_dict)
        self.rec_model.eval()

    @staticmethod
    def get_config(file_path): #загрузка параметров НС (распознавателя) из YAML файла
        with open(file_path, 'r', encoding="utf8") as stream:
            opt = yaml.safe_load(stream)
        opt = AttrDict(opt)
        opt.character = opt.number + opt.symbol + opt.lang_char
        
        return opt

    @staticmethod
    def prepare_dict(state_dict): #приведение модели к стандартному виду
        copy_state_dict = copy.deepcopy(state_dict)
        for key in copy_state_dict.keys():
            if key.startswith('module'):
                new_key = key.removeprefix('module.')
                state_dict[new_key] = state_dict.pop(key)

        return state_dict

    @staticmethod
    def custom_mean(x):
        return x.prod()**(2.0/np.sqrt(len(x)))

    @staticmethod
    def crop(image, box):
        return np.array(image.crop(box = box))

    @staticmethod
    def reformat_input(image): #наложение фильтров на исходное изображени (ч/б, бинаризация)
        image_array = np.array(image)
        img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_cv_th = cv2.adaptiveThreshold(img_cv_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

        return img_cv_grey, img_cv_th
    
    #обнаружение областей, содержащих показания
    def detect(self, image, threshold=0.8):
        with torch.no_grad():
            inputs = self.det_img_processor(images=image, return_tensors="pt")
            outputs = self.det_model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.det_img_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
                
        draw = ImageDraw.Draw(image)
        value_b = None
        value_a = None

        #расшифровка результата 
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label.item() in [0,1]:
                box = tuple(round(i, 2) for i in box.tolist())
                """ print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}") """
                
                x, y, x2, y2 = box
                
                draw.rectangle((x-5, y-5, x2+5, y2+5), outline="red", width=2)
                draw.text((x-15, y-15), self.det_model.config.id2label[label.item()], fill="white")

                if label.item() == 0:
                    value_a = self.crop(image, (x-5, y-5, x2+5, y2+5))        
                elif label.item() == 1:
                    value_b = self.crop(image, (x-5, y-5, x2+5, y2+5))

        return image, value_b, value_a

    #распознание показаний
    def recognize_values(self, image):
        text_for_pred = torch.LongTensor(self.opt.batch_size, self.opt.batch_max_length + 1).fill_(0).to(device)

        all_results = []
        for image in self.reformat_input(image): #получаем предсказания для каждого отформатированного изображения отдельно
                
            #отравляем изображение в НС
            with torch.no_grad():       
                transform = transforms.ToTensor()
                inputs = transform(image).unsqueeze(0)
                preds = self.rec_model(inputs, text_for_pred, is_train=False ) 

            #обработка результатов работы распознавателя
            preds_size = torch.IntTensor([preds.size(1)] * self.opt.batch_size)
            preds_prob = F.softmax(preds, dim=2)
            preds_prob = preds_prob.cpu().detach().numpy()

            pred_norm = preds_prob.sum(axis=2)
            preds_prob = preds_prob/np.expand_dims(pred_norm, axis=-1) 
            preds_prob = torch.from_numpy(preds_prob).float().to(device)

            converter = AttnLabelConverter(self.opt.character)
            _, preds_index = preds_prob.max(2)
            preds_index = preds_index.view(-1)
            preds_str = converter.decode(preds_index.data.cpu().detach().numpy(), preds_size.data)

            preds_prob = preds_prob.cpu().detach().numpy()
            values = preds_prob.max(axis=2)
            indices = preds_prob.argmax(axis=2)
            preds_max_prob = []
            for v,i in zip(values, indices):
                max_probs = v[i!=0]
                if len(max_probs)>0:
                    preds_max_prob.append(max_probs)
                else:
                    preds_max_prob.append(np.array([0]))
            result = []
            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                confidence_score = self.custom_mean(pred_max_prob)
                result.append([pred, confidence_score])
            indx = result[0][0].find('[s]')
            all_results.append((result[0][0][:indx], result[0][1]))
        best_match = max(all_results, key = lambda x: x[1]) #выбираем предсказание с наилучшим результатом

        return best_match[0]
