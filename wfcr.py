import torch
import numpy as np
import cv2
import yaml
import copy
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F

from PIL import Image, ImageDraw,  ImageOps
from scipy.stats import linregress

from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from recognition.utils import AttrDict, AttnLabelConverter
from recognition.model import Model


class WFCR():
    """
    Данный класс при инициализации подгружает модель детектора и распознавателя,
    которые хранятся в папках на локальном компьютере. Путь к ним передается в качестве аргумента 
    по умолчанию:
        путь к детектору - 'maskrcnn/maskrcnn_resnet50_fpn_v2.pth'
        путь к распознавателю - recognition\wfcr_model.pth)
        путь к настройкам модели - 'recognition\wfcr_model_config.yaml'

    Методы класса:
    detect: производит детецию областей, содержащих показания счетчика потребления воды. 
            Отдельно определяет области с показаниями до запятой и после запятой. 
            Возможно задать точность ниже которой область будет считаться 
            не распознанной (по умолчанию - 80%).
            Принимает в качестве аргументов:
                - изображение (путь к расположению на диске, байты или массив)
                - размер изображения в виде кортежа (W,H) -необходим в случае передачи изображения в виде байтов иначе None
                - трешхолд

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
            det_path = 'maskrcnn/maskrcnn_resnet50_fpn_v2.pth', 
            rec_path = 'recognition/wfcr_model.pth',
            rec_conf_path = 'recognition/wfcr_model_config.yaml'
        ):
        #выбор CPU или GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32

        #загрузка модели детектора
        self.det_model = maskrcnn_resnet50_fpn_v2(weights=None)
        
        in_features_box = self.det_model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = self.det_model.roi_heads.mask_predictor.conv5_mask.in_channels
        dim_reduced = self.det_model.roi_heads.mask_predictor.conv5_mask.out_channels

        #pамена головы модели (bbox)
        self.det_model.roi_heads.box_predictor = FastRCNNPredictor(in_channels=in_features_box, num_classes=2)

        #замена головы модели (mask)
        self.det_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_channels=in_features_mask, dim_reduced=dim_reduced, num_classes=2)

        self.det_model.to(device=self.device, dtype=self.dtype)

        #добавление атрибутов для последующего использования
        self.det_model.device = self.device
        self.det_model.name = det_path.split('/')[1].split('.')[0]

        self.det_model.load_state_dict(torch.load(det_path, map_location=self.device))
        self.det_model.eval()

        #загрузка модели распознавателя
        self.opt = self.get_config(rec_conf_path)
        self.state_dict = self.prepare_dict(torch.load(rec_path, map_location=self.device))
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

    def reformat_input(self, image): #наложение фильтров на исходное изображени (ч/б, бинаризация)
        image_array = np.array(image)
        image_array = image_array[:, :, ::-1].copy()
        img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img_cv_sh = cv2.adaptiveThreshold(img_cv_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        img_sh = cv2.filter2D(img_cv_grey, -1, kernel)

        img_cv_grey_rot = self.rotate(img_cv_grey, 180, PIL=False)
        img_sh_rot = self.rotate(img_sh, 180, PIL=False)

        return img_cv_grey, img_sh, img_cv_grey_rot, img_sh_rot
    
    #чтение изображения в зависимости от формата
    @staticmethod
    def read_image(image, size=None):
        if type(image) == str: #если дан путь к изображению на локальном диске
            img = Image.open(image)
            img = ImageOps.exif_transpose(img)      
        elif type(image) == bytes: #если переданы байты
            if size:
                img = Image.frombytes('RGB', size, image, 'raw')
            else:
                raise AttributeError('Attribute "size" not found')
        elif type(image) == np.ndarray: #если передан массив
            if len(image.shape) == 2:  #если ч/б
                img = Image.fromarray(image, 'RGB')
            elif len(image.shape) == 3 and image.shape[2] == 3: #если RGB
                img = Image.fromarray(image, 'RGB')
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
                img = Image.fromarray(image[:, :, :3], 'RGB')
        elif type(image).__module__.split('.')[0] == "PIL":
            img = image    
        return img

    #изменение размера
    @staticmethod
    def resize_img(img, target_sz = 512, divisor = 32):
        
        min_dim = np.argmin(img.size)
        max_dim = np.argmax(img.size)
        
        ratio = min(img.size)/target_sz
        new_sz = []
        new_sz.insert(min_dim, target_sz)
        new_sz.insert(max_dim, int(max(img.size)/ratio))
        
        img = img.resize(new_sz)

        if divisor > 0:
            src_w, src_h = img.size
            width = src_w if src_w%divisor==0 else src_w - src_w%divisor
            height = src_h if src_h%divisor==0 else src_h - src_h%divisor
            img = img.crop(box=(0, 0, width, height))
        
        return img
    
    @staticmethod
    def draw_mask(image, mask_generated) :
        masked_image = image.copy()

        masked_image = np.where(mask_generated.astype(int)[...,None],
                                np.array([0,255,0], dtype='uint8'),
                                masked_image)

        masked_image = masked_image.astype(np.uint8)

        return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)

    #разворот изображения с помощью библиотеки OpenCv
    @staticmethod
    def rotate(image, angle, scale = 1, PIL=True):
        image_array = np.asarray(image, dtype="uint8")
        
        height, width = image_array.shape[:2] # image shape has 3 dimensions
        image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
        
        rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
        
        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0,0]) 
        abs_sin = abs(rotation_mat[0,1])
        
        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        
        # subtract old image center (bringing image back to origo) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w/2 - image_center[0]
        rotation_mat[1, 2] += bound_h/2 - image_center[1]
        
        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(image_array, rotation_mat, (bound_w, bound_h))
        
        if PIL:
            rotated_mat = Image.fromarray(rotated_mat)

        return rotated_mat

    #обнаружение областей, содержащих показания
    def detect(self, input_image, size=None, threshold=0.8):
        image = self.read_image(input_image, size)
        image_res = self.resize_img(image, target_sz=512, divisor=1)
        image_bbox = image_res.copy()
        image_rot = image_res.copy()
        image_masked = image_res.copy()

        self.det_model.to(self.device)
        input_tensor = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])(image_res)[None].to(self.device)

        #получаем предсказание
        with torch.no_grad():
            model_output = self.det_model(input_tensor)

        #отбрасываем области обнаруженные с уверенностью ниже 80%
        
        score = np.asarray(model_output[0]['scores']).tolist()

        #print(f"scores- {score}\nbboxes-{bboxes}\nmask-{masks}")

        if score:
            print(f"Найдено {score}")
            bboxes = np.array(model_output[0]['boxes']).tolist()
            masks = np.array(model_output[0]['masks'])[0]
            max_score = max(score)
            if max_score >= threshold:
                indx = score.index(max_score)
                box = bboxes[indx]
                x1, y1, x2, y2 = box
                rate = (x2 - x1) / (y2 - y1)
                print(rate)

                #print(rate)
                if rate >= 0.2 and rate <= 0.3:
                    image_res = self.rotate(image_res, 90)
                    print("Поворачиваем изображение на 90")
                elif rate > 0.3 and rate <= 0.75:
                    image_res = self.rotate(image_res, 65)
                    print("Поворачиваем изображение на 65")
                elif rate > 0.75 and rate <= 1:
                    image_res = self.rotate(image_res, 45)
                    print("Поворачиваем изображение на 45")
                else:
                    mask = masks[indx]
                    mask[mask<0.5] = 0
                    mask = mask.astype(bool)

                    image_array = np.asarray(image_res)[:,:,-1]

                    X = []
                    Y = []
                    for coord_y, row in enumerate(image_array):
                        for coord_x, pixel in enumerate(row):
                            if mask[coord_y,coord_x]:
                                #print(f"coord: x-{coord_x}, y-{coord_y}")
                                X.append(coord_x)
                                Y.append(coord_y)

                    res = linregress(X, Y)
                    slope = res.slope
                    intercept = res.intercept
                    reg_line_y1 = intercept + slope*x1
                    reg_line_y2 = intercept + slope*x2

                    angle = int(np.arctan2(reg_line_y2 - reg_line_y1, x2-x1)*180/np.pi)
                                    
                    #рисуем bbox вокруг найденной области
                    draw = ImageDraw.Draw(image_bbox)                   
                    draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
                    draw.text((x1, y1-10), 'Value', fill="white")
                    
                    #выделяем область со значениями
                    image_bbox = self.draw_mask(np.asarray(image_bbox), mask)
                    image_bbox = Image.fromarray(image_bbox)
                                        
                    print(f"Изображение будет развернуто на {angle}°")
                    #разворачиваем и обрезаем по bbox
                    image_rot = image_rot.crop((x1, y1 - 5, x2, y2 + 5))
                    image_rot = self.rotate(image_rot, angle)

                    #разворачиваем накладываем маску и обрезаем по bbox
                    image_masked = np.where(~mask[...,None], np.array([0,0,0], dtype='uint8'), image_masked)
                    image_masked = Image.fromarray(image_masked)
                    image_masked = image_masked.crop((x1, y1, x2, y2))
                    image_masked = self.rotate(image_masked, angle)

                    if abs(angle) > 15:
                        w,h = image_rot.size
                        delta_y = h/3
                        image_rot = image_rot.crop((0, delta_y, w, h-delta_y))
                        image_masked = image_masked.crop((0, delta_y, w, h-delta_y))
                    return image_bbox, image_rot, image_masked

                print(f"Изображение будет повторно отправлено в детектор")        
                return self.detect(image_res)
            
        print(f"Ничего не найдено")
        return None, None, None
            

    #распознание показаний
    def recognize_values(self, image):
        text_for_pred = torch.LongTensor(self.opt.batch_size, self.opt.batch_max_length + 1).fill_(0).to(self.device)

        all_results = []
        for image in self.reformat_input(image): #получаем предсказания для каждого отформатированного изображения отдельно
            #отравляем изображение в НС
            with torch.no_grad():       
                transform = transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)])
                inputs = transform(image[...,None]).unsqueeze(0)     
                preds = self.rec_model(inputs, text_for_pred, is_train=False ) 

            #обработка результатов работы распознавателя
            preds_size = torch.IntTensor([preds.size(1)] * self.opt.batch_size)
            preds_prob = F.softmax(preds, dim=2)
            preds_prob = preds_prob.cpu().detach().numpy()

            pred_norm = preds_prob.sum(axis=2)
            preds_prob = preds_prob/np.expand_dims(pred_norm, axis=-1) 
            preds_prob = torch.from_numpy(preds_prob).float().to(self.device)

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
        indx = all_results.index(best_match)

        return best_match[0], indx
