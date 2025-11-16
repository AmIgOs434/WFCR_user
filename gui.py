import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

from PIL import ImageTk


class Window(tk.Tk):
    def __init__(self, func):
        super().__init__()
        
        self.title('Автоматическое распознавание показаний счетчиков')  #заголовок рабочего окна
        self.resizable(width=False, height=False) #отключение возможности растягивать окно 

        #создание родительского фрэйма - frame 1
        frame1 = tk.Frame(self, borderwidth=5, relief='sunken')
        frame1.grid(column=0, row=0)
        
        #создание поля для ввода данных
        lbl_get_path = tk.Label(frame1, text='Укажите путь к фотографии', font=('Calibri', 9))
        ent_get_path = ttk.Entry(frame1, width=50)

        lbl_get_path.grid(column=0, row=0,)
        ent_get_path.grid(column=0, row=1,)

        def open_file():
            filepath = filedialog.askopenfilename()
            if filepath != "":
                ent_get_path.delete(0,tk.END)
                ent_get_path.insert(0,filepath)
        #создание кнопок
        btn1 = tk.Button(frame1, text='Запуск', command=lambda: func(ent_get_path.get()))
        btn2 = tk.Button(frame1, text='Выход', command=self.quit)
        btn3 = tk.Button(frame1, text='Выбрать файл', command=open_file)
        btn1.grid(column=0, row=4, padx=5)
        btn2.grid(column=0, row=5, padx=5)
        btn3.grid(column=0, row=3, padx=5)


    def result_window(self, image, value, qrCode, win_width = 500):
        window = tk.Toplevel(self)
        new_width = win_width
        width,length = image.size
        ratio = new_width/width
        new_length = length*ratio
        img = image.resize((int(new_width), int(new_length)))
        img = ImageTk.PhotoImage(img)

        #создание фрэйма
        frame_ph = tk.Frame(window, borderwidth=5, relief='sunken')
        frame_ph.pack()

        #создание фото и надписи
        lbl_photo = tk.Label(frame_ph, image=img)
        lbl_photo.image = img
        lbl_value = tk.Label(window, text=f'Показания счетчика: {value}', font=('Calibri', 16))
        lbl_qrcode = tk.Label(window, text=f'{qrCode}', font=('Calibri', 16))
        lbl_photo.pack()
        lbl_value.pack()
        lbl_qrcode.pack()