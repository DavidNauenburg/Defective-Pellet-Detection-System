#!/usr/bin/env python3

from datetime import date, datetime
from PIL import Image, ImageTk
from tkinter import ttk
from ImageAnalysis import ImageAnalysis
import numpy as np
import os
import csv
import tkinter as tk
import cv2
import time
import threading
import torch
import usb.core

LABEL = "bad"
LARGEFONT =("Verdana", 35)
MED_FONT = ("Verdana", 10)
SPECIAL_CHARS = "\!@\#][}{:;'$%^&*)(_+=,><.?`~/"
  
class tkinterApp(tk.Tk):
     
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
         
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Defective Pellet Detection System")
         
        # creating a container
        self.container = tk.Frame(self) 
        self.container.pack(side = "top", fill = "both", expand = True)
  
        self.container.grid_rowconfigure(0, weight = 1)
        self.container.grid_columnconfigure(0, weight = 1)

        #self.video_sources = [('Camera 1', 0)]
        self.video_sources = []
        self.get_camera_indices()
        self.container.tk.call(
            "source",
            "C:/Users/david.nauenburg/Desktop/Omnibus_22Aug2022/azure.tcl"
        )
        #self.container.tk.call("source","azure.tcl")
        self.container.tk.call("set_theme", "dark")

        # Creates export folder to store images, csv files
        # and interactive html figures
        self.hold = date.today().strftime("%d_%b_%Y_export_")
        self.temp = time.strftime('%I_%M_%p')
        self.combined = self.hold + self.temp
        self.folder_path = f"{os.getcwd()}\{self.combined}"
        if os.path.exists(self.folder_path) == False:
            os.mkdir(self.folder_path)

        # Initializing frames to an empty array
        # This will hold the pages of the application
        self.frames = {} 
  
        # Iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, RealTimeDetectionPage, Page2):
  
            frame = F(self.container, self)
  
            # Initializing frame of that object from
            # Login, RealTimeDetectionPage, AnalysisPage 
            # respectively with for loop
            self.frames[F] = frame
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(StartPage)

        # This will hold the tkCamera objects
        self.vids = []
        # Names of photos in export folder
        self.photo_names = set()
        # This array will allow single images to persist in memory
        self.single_imgs = set()

        # Create a tkCamera object for each camera connected
        # The streams will be displayed in the RealTimeDetectionPage
        columns = 2
        for number, source in enumerate(self.video_sources):
            text, stream = source
            y = number % columns # x
            x = number // columns # y
            vid = tkCamera(
                self.frames[RealTimeDetectionPage],
                self.folder_path, 
                text,
                stream, 
                400, # 640 
                300, # 480
                x, 
                y
            )
            self.vids.append(vid)

        # Sizegrip at bottom right of application
        self.container.sizegrip = ttk.Sizegrip(self.container)
        self.container.sizegrip.grid(
            row = 100, 
            column = 100, 
            padx = (0, 5), 
            pady = (0, 5)
        )

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def make_responsive(self, grid_size, page):
        """
        This method makes each page of the app responsive
        to window size changes.
        """
        num_rows = grid_size[1]
        num_cols = grid_size[0]
        for row in range(0, num_rows + 1):
            #self.container.columnconfigure(index = index, weight = 1)
            page.rowconfigure(index = row, weight = 1)
        for col in range(0, num_cols + 1):
            page.columnconfigure(index = col, weight = 1)


    def get_camera_indices(self):
        """
        This method checks which usb ports have cameras connected.
        The indices are appended to video_sources which will be
        used to create tkCameras which will display a livestream
        and model runing on top of it. 
        """
        num_usb = len(list(usb.core.find(find_all = True)))
        for port_id in range(num_usb):
            cap = cv2.VideoCapture(port_id)
            if cap.isOpened():
                self.video_sources.append(
                    (f"Camera {port_id}", port_id)
                )
                cap.release()

    def equal_weight(self, grid_size, frame):
        """
        The method ensures the rows and columns of each frame have equal
        weight and adjusts to the correct proportions when the window
        size changes 
        """
        row_index = grid_size[1]
        col_index = grid_size[0]
        if row_index == col_index:
            for a in range(0, row_index):
                frame.grid_rowconfigure(a, weight = 1)
                frame.grid_columnconfigure(a, weight = 1)
        else:
            for i in range(0,row_index):
                frame.grid_rowconfigure(i, weight = 1)
        
            for j in range(0, col_index):
                frame.grid_columnconfigure(j, weight = 1)

    def check_img_exists(self, img):
        for img_obj in self.single_imgs:
            if img == img_obj.image:
                return True
        return False

    def update_photo_names(self):
        self.photo_names.clear()
        for photo in os.listdir(self.folder_path):
            if photo.endswith('.jpg'):
                self.photo_names.add(photo)

    def delete_all_images(self):
        single_img_list = list(self.single_imgs)
        self.single_imgs.clear()
        for img in single_img_list:
            img.delete_object()
            SingleImage.counter -= 1

    def update_page_two(self):
        if SingleImage.counter > 0:
            self.delete_all_images()
        self.update_photo_names()
        columns = 2
        num_jpg = len(self.photo_names)
        index = 0

        if SingleImage.counter == num_jpg:
            return
        elif num_jpg > SingleImage.counter:
            for img in self.photo_names:
                y = index % columns # x
                x = index // columns # y
                path_to_image = f"{self.folder_path}\{img}"
                single_img = SingleImage(
                    self.frames[Page2], 
                    path_to_image, 
                    x, 
                    y,
                    index
                )
                index += 1
                # This allows the objects to persist in memory,
                # otherwise the images would be garbage collected.
                self.single_imgs.add(single_img)


    # Display the page that was passed as a parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

        if cont == Page2:
            self.update_page_two()

    # Destroy the TkinterApp on closing
    def on_closing(self, event=None):
        for source in self.vids:
            source.tk_vid.running = False
            source.tk_vid.thread.join()

        if source.tk_vid.my_vid.isOpened():
            source.tk_vid.my_vid.release()
        self.destroy()
    
# First page to appear on start up
class StartPage(tk.Frame, tkinterApp): 
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # User list
        OpA_txt = open(
            "C:/Users/david.nauenburg/Desktop/Omnibus_22Aug2022/List of OpAs", 
            "r"
        )
        self.Op_As = OpA_txt.read()
        self.OpA_list = self.Op_As.split(',')
        OpA_txt.close()

        self.title_frame = ttk.Frame(
            self,
            padding = (10, 10)
        )
        self.title_frame.grid(
            row = 0,
            column = 0,
            padx = (10, 10),
            pady = (10, 10),
            sticky = "nsew"
        )

        self.label = ttk.Label(
            self.title_frame, 
            text ="Login Information",
            font = LARGEFONT,
        )
        self.label.grid(
            row = 0,
            column = 0,
            padx = (10, 10),
            pady = (10, 10),
            sticky = "nsew",
        )

        # Give equal weight to rows and cols of title_frame
        # Access equal_frame
        super(StartPage, self).equal_weight(
            self.title_frame.grid_size(),
            frame = self.title_frame
        )

        # Create a Frame for user input
        self.user_frame = ttk.LabelFrame(
            self, 
            text = "User Input", 
            padding = (20, 10)
        )
        self.user_frame.grid(
            row = 1, 
            column = 0, 
            padx = (20, 10), 
            pady = (20, 10), 
            sticky = "nsew"
        )

        # User login label
        self.user_login = ttk.Label(
            self.user_frame, 
            text = "Login Op A:", 
            font = MED_FONT
        )
        self.user_login.grid(
            row = 0, 
            column = 0, 
            padx = (0,10), 
            pady = 10, 
            sticky = "nsew"
        )

        # User Combobox
        self.combobox = ttk.Combobox(
            self.user_frame, 
            values=self.OpA_list
        )
        self.combobox.current(0)
        self.combobox.grid(
            row = 0, 
            column = 1, 
            padx = 5, 
            pady = 10, 
            sticky = "nsew"
        )

        # Date entry label
        self.date_label = ttk.Label(
            self.user_frame, 
            text = "Date:", 
            font = MED_FONT
        )
        self.date_label.grid(
            row = 0, 
            column = 2, 
            padx = (110,0), 
            pady = 10, 
            sticky = "nsew"
        )

        # Date entry field
        self.today = date.today()
        self.today = self.today.strftime("%d %b %Y")
        self.date_entry = ttk.Entry(self.user_frame)
        self.date_entry.insert(0, self.today)
        self.date_entry.grid(
            row = 0, 
            column = 2, 
            padx = (170,0), 
            pady = 10, 
            sticky = "nsew"
        )

        # Lot number label
        self.lot_label = ttk.Label(
            self.user_frame, 
            text = "Lot Name:", 
            font = MED_FONT
        )
        self.lot_label.grid(
            row = 1, 
            column = 0, 
            padx = (0,10), 
            pady = 10, 
            sticky = "nsew"
        )

        # Lot entry field
        self.lot_entry = ttk.Entry(self.user_frame)
        self.lot_entry.insert(0, "Cov-Bi-YYYYMMDD-OP-T#K-A")
        self.lot_entry.grid(
            row = 1, 
            column = 1, 
            padx = 5, 
            pady = 10, 
            sticky = "nsew"
        )

        # Time label
        self.time_label = ttk.Label(
            self.user_frame, 
            text = "Time:", 
            font = MED_FONT
        )
        self.time_label.grid(
            row = 1, 
            column = 2, 
            padx = (110,0), 
            pady = 10, 
            sticky = "nsew"
        )

        # Time entry field
        self.local_time = time.localtime()
        self.current_time = time.strftime("%I:%M %p", self.local_time)
        self.time_entry = ttk.Entry(self.user_frame)
        self.time_entry.insert(0, self.current_time) 
        self.time_entry.grid(
            row = 1, 
            column = 2, 
            padx = (170,0), 
            pady= 10, 
            sticky = "nsew"
        )

        # Lyo number label
        self.lyo_label = ttk.Label(
            self.user_frame, 
            text = "Lyo Number:", 
            font = MED_FONT
        )
        self.lyo_label.grid(
            row = 2, 
            column = 0, 
            padx = (0,10), 
            pady = 10, 
            sticky = "nsew"
        )

        # Lyo entry field
        self.lyo_entry = ttk.Entry(self.user_frame)
        self.lyo_entry.insert(0, "")
        self.lyo_entry.grid(
            row = 2, 
            column = 1, 
            padx = 5, 
            pady = 10, 
            sticky = "nsew"
        )

        # Op B label
        self.opB_label = ttk.Label(
            self.user_frame, 
            text = "Login Op B:", 
            font = MED_FONT
        )
        self.opB_label.grid(
            row = 2, 
            column = 2, 
            padx = (75,0), 
            pady = 10, 
            sticky = "nsew"
        )

 
        # OP B Combobox
        self.combobox2 = ttk.Combobox(
            self.user_frame, 
            values=self.OpA_list
        )
        self.combobox2.current(0)
        self.combobox2.grid(
            row = 2, 
            column = 2, 
            padx = (170,0), 
            pady = 10, 
            sticky = "nsew"
        )

        # Give equal weight to rows and cols so the frame adjusts
        # properly when the window is resized
        # Access equal_weight function from parent class,
        # tkinterApp
        super(StartPage, self).equal_weight(
            self.user_frame.grid_size(),
            frame = self.user_frame
        )

        self.btn_frame1 = ttk.Frame(
            self,
            padding = (10, 10)
        )
        self.btn_frame1.grid(
            row = 100,
            column = 0,
            padx = (10, 10),
            pady = (10, 10),
            sticky = "nsew"
        )
  
        # Button to switch to Real Time Object Detection Page
        self.button1 = ttk.Button(
            self.btn_frame1,
            text = "Real Time Object Detection",
            style = "Accent.TButton",
            command = lambda : controller.show_frame(RealTimeDetectionPage)
        )
        self.button1.grid(
            row = 0, 
            column = 0,
            padx = (10, 10), # (10, 175)
            pady = 10
        )
        self.btn_frame2 = ttk.Frame(
            self,
            padding = (10, 10)
        )
        self.btn_frame2.grid(
            row = 100,
            column = self.grid_size()[0],
            padx = (10, 10),
            pady = (10, 10)
        )
        # Button to show page 2
        self.button2 = ttk.Button(
            self.btn_frame2, 
            text = "Image Analysis",
            style = "Accent.TButton",
            command = lambda : controller.show_frame(Page2)
        )
        self.button2.grid(
            row = 0,
            column = 0, 
            padx = (10, 10), # (175, 10)
            pady = 10
        )

        # Give equal weight to rows and cols so the frame adjusts
        # properly when the window is resized
        super(StartPage, self).equal_weight(
            self.grid_size(),
            frame = self.user_frame
        )


        # Make StartPage responsive to window adjustments
        super(StartPage, self).make_responsive( 
            self.grid_size(),
            self
        )
  
  
# Second page the user sees
class RealTimeDetectionPage(tk.Frame, tkinterApp): 
     
    def __init__(self, parent, controller):
         
        tk.Frame.__init__(self, parent)

        self.label = ttk.Label(
            self,
            text = "Real Time Object Detection",
            font = LARGEFONT
        )
        self.label.grid(
            row = 0,
            column = 0,
            padx = 10,
            pady = 10
        )

        self.btn_frame1 = ttk.Frame(
            self,
            padding = (10, 10)
        )
        self.btn_frame1.grid(
            row = 100,
            column = 0,
            padx = (10, 10),
            pady = (10, 10),
            sticky = "nsew"
        )
  
        self.button1 = ttk.Button(
            self.btn_frame1,
            text = "Login Information",
            style = "Accent.TButton",
            command = lambda : controller.show_frame(StartPage)
        )
        self.button1.grid(
            row = 0,
            column = 0,
            padx = 10,
            pady = 10
        )

        self.btn_frame2 = ttk.Frame(
            self,
            padding = (10, 10)
        )
        self.btn_frame2.grid(
            row = 100,
            column = self.grid_size()[0],
            padx = (10, 10),
            pady = (10, 10),
            sticky = "nsew"
        )

        self.button2 = ttk.Button(
            self.btn_frame2,
            text = "Image Analysis",
            style = "Accent.TButton",
            command = lambda : controller.show_frame(Page2)
        )
        self.button2.grid(
            row = 0,
            column = 0,
            padx = 10,
            pady = 10
        )
        super(RealTimeDetectionPage, self).equal_weight( 
            self.grid_size(),
            self
        )

        super(RealTimeDetectionPage, self).make_responsive(
            self.grid_size(),
            self
        )

# The third page the user sees
class Page2(tk.Frame, tkinterApp): 
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.label = ttk.Label(
            self, 
            text = "Image Analysis", 
            font = LARGEFONT
        )
        self.label.grid(
            row = 0, 
            column = 0, 
            padx = 10, 
            pady = 10
        )

        self.btn_frame1 = ttk.Frame(
            self,
            padding = (10, 10)
        )
        self.btn_frame1.grid(
            row = 100,
            column = 0,
            padx = (10, 10),
            pady = (10, 10)
        )
        self.button1 = ttk.Button(
            self.btn_frame1,
            text = "Real Time Object Detection",
            style = "Accent.TButton",
            command = lambda : controller.show_frame(RealTimeDetectionPage)
        )
        self.button1.grid(
            row = 0,
            column = 0,
            padx = 10,
            pady = 10
        )

        self.btn_frame2 = ttk.Frame(
            self,
            padding = (10, 10)
        )
        self.btn_frame2.grid(
            row = 100,
            column = self.grid_size()[0],
            padx = (10, 10),
            pady = (10, 10)
        )
        self.button2 = ttk.Button(
            self.btn_frame2,
            text = "Login Information",
            style = "Accent.TButton",
            command = lambda : controller.show_frame(StartPage)
        )
        self.button2.grid(
            row = 0,
            column = 0,
            padx = 10,
            pady = 10
        )

        super(Page2, self).equal_weight(
            self.grid_size(),
            self
        )

        super(Page2, self).make_responsive(
            self.grid_size(),
            self
        )

class SingleImage(tk.Frame, tkinterApp):
    # Keep track of number of SingleImages that have been created
    counter = 0
    analysis_counter = 1
    def __init__(
        self, 
        parent, 
        image, 
        x = 0, 
        y = 0,
        index = 0
    ):
        SingleImage.counter += 1
        self.parent = parent
        self.image = image
        self.my_img = (Image.open(self.image))
        self.new_img = ImageTk.PhotoImage(self.my_img)
        self.index = index
        self.ref_width = 0.0
        self.widget_list = []

        # Create a Frame where the canvas and buttons will go
        self.image_frame = ttk.LabelFrame(
            self.parent,
            text = image, # FIXME: Edit image text name
            padding = (10, 10)
        )
        self.image_frame.grid(
            row = x+1,
            column = y,
            padx = (10, 10),
            pady = (10, 10),
            sticky = "nsew"
        )
        self.add_to_widget_list(self.image_frame)

        # This is where the image will be seen
        self.img_canvas = tk.Canvas(
            self.image_frame, 
            width = 400, 
            height = 300
        )
        self.img_canvas.grid(
            row = x, # y+1, or x+2
            column = y, # x
            padx = 10,
            pady = 10
        )
        self.add_to_widget_list(self.img_canvas)

        # Show image in canvas
        self.img_canvas.create_image(0, 0, image = self.new_img, anchor = 'nw')

        # Give equal weight to rows and cols of image_frame
        super(SingleImage, self).equal_weight(
            self.image_frame.grid_size(),
            self.image_frame
        )

        # Frame for buttons
        self.btn_frame = ttk.LabelFrame(
            self.image_frame,
            text = "Controls",
            padding = (10, 10)
        )
        self.btn_frame.grid(
            row = x+1,
            column = y,
            padx = (10, 10),
            pady = (10, 10)
        )
        self.add_to_widget_list(self.btn_frame)

        self.ref_label = ttk.Label(
            self.btn_frame,
            text = "Reference Width (mm):",
            padding = (10, 10)
        )
        self.ref_label.grid(
            row = 0,
            column = 0,
            padx = (10, 10),
            pady = (10, 10)
        )
        self.add_to_widget_list(self.ref_label)

        self.ref_entry = ttk.Entry(self.btn_frame)
        self.ref_entry.grid(
            row = 0, 
            column = 1, 
            padx = (10, 10), 
            pady = (10, 10), 
            sticky = "nsew"
        )
        self.add_to_widget_list(self.ref_entry)

        self.btn_analysis = ttk.Button(
            self.btn_frame,
            text = "Image Analysis",
            padding = (10, 10),
            style = "Accent.TButton", 
            command = self.image_analysis
        )
        self.btn_analysis.grid(
            row = 0,
            column = 2,
            padx = (10, 10),
            pady = (10, 10)
        )
        self.add_to_widget_list(self.btn_analysis)

        # Give equal weight to rows and cols of btn_frame
        super(SingleImage, self).equal_weight(
            self.btn_frame.grid_size(),
            self.btn_frame
        )

    def add_to_widget_list(self, widget):
        self.widget_list.append(widget)

    def delete_object(self):
        for widget in self.widget_list:
            widget.grid_forget()
            
        for widget in self.widget_list:
            widget.destroy()

    def check_ref_width(self, width):
        if width <= 0 or width > 4 :
            return False
        return True

    def image_analysis(self):
        try:
            if (self.check_ref_width(float(self.ref_entry.get()))):
                self.ref_width = float(self.ref_entry.get())
            else:
                raise ValueError(
                    "Reference width should be greater than 0 and less than 4."
                )
            temp = os.path.dirname(self.image)
            self.analysis = ImageAnalysis(
                self.image, 
                self.ref_width, 
                SingleImage.analysis_counter
            )
            output_path = (
                temp + f"\output_{SingleImage.analysis_counter}.csv"
            )
            self.analysis.df.to_csv(
                output_path, 
                sep = '\t', 
                na_rep = 'NaN', 
                index = False
            )
            SingleImage.analysis_counter += 1
        except FileNotFoundError:
            print(f"{self.image} not found. Check folder.")



class MyVideoCapture:
    def __init__(
        self, 
        video_source = 0, 
        width = None, 
        height = None, 
        fps = None
    ):
        self.video_source = video_source
        self.width = width
        self.height = height
        self.fps = fps
        self.model_flag = True

        # Load custom trained model
        self.model = torch.hub.load(
            'C:/Users/david.nauenburg/Desktop/Omnibus_22Aug2022/yolov5', 
            'custom', 
            path='C:/Users/david.nauenburg/Desktop/Omnibus_22Aug2022/best.pt', 
            source='local'
        )
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        # Open the video source
        self.my_vid = cv2.VideoCapture(video_source)
        if not self.my_vid.isOpened():
            raise ValueError(
                "[MyVideoCapture] Unable to open video source", video_source
            )

        # Get video source width and height
        if not self.width:
            # convert float to int
            self.width = int(self.my_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        if not self.height:
            # convert float to int
            self.height = int(self.my_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not self.fps:
            # convert float to int
            self.fps = int(self.my_vid.get(cv2.CAP_PROP_FPS))

        # default value at start        
        self.ret = False
        self.frame = None

        # start thread
        self.running = True
        self.thread = threading.Thread(target=self.process)
        self.thread.start()

    def process(self):
        while self.running:
            ret, frame = self.my_vid.read()

            if ret:
                if self.model_flag:
                    results = self.score_frame(frame)
                    frame = self.plot_boxes(results, frame)

                # process image
                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                print('[MyVideoCapture] stream end:', self.video_source)
                # TODO: reopen stream
                self.running = False
                break
                
            # assign new frame
            self.ret = ret
            self.frame = frame
            
            # sleep for next frame
            time.sleep(1/self.fps)
    
    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 
        model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in 
                 the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels = results.xyxyn[0][:, -1] 
        coord = results.xyxyn[0][:, :-1]

        return labels, coord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding 
        boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by 
                        model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, coord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            # Ignore good pellets
            if self.class_to_label(labels[i]) != LABEL:
                pass
            else: # Draw bounding box around bad pellets
                row = coord[i]
                if row[4] >= 0.3:
                    x1 = int(row[0] * x_shape)
                    y1 = int(row[1] * y_shape)
                    x2 = int(row[2] * x_shape)
                    y2 = int(row[3] * y_shape)
                    bgr = (0, 0, 255) # red
                    cv2.rectangle(
                        frame, 
                        (x1, y1), 
                        (x2, y2), 
                        bgr, 
                        2
                    )
                    cv2.putText(
                        frame, 
                        self.class_to_label(labels[i]), 
                        (x1, y1), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        bgr, 
                        2
                    )

        return frame

    def get_frame(self):
        return self.ret, self.frame
    
    # Release the video source when the object is destroyed
    def __del__(self):
        # stop thread
        if self.running:
            self.running = False
            self.thread.join()

        # release stream
        if self.my_vid.isOpened():
            self.my_vid.release()


class tkCamera(tk.Frame, tkinterApp):
    def __init__(
        self,
        parent, 
        folder_path,
        text = "", 
        video_source = 0, 
        width = None, 
        height = None, 
        x = 0, 
        y = 0
    ):
        super().__init__(parent)
        self.parent = parent
        self.folder_path = folder_path

        self.video_source = video_source
        self.tk_vid = MyVideoCapture(self.video_source, width, height)

        # Create a Frame where the canvases and buttons will go
        self.camera_frame = ttk.LabelFrame(
            self.parent,
            text = text,
            padding = (10, 10)
        )
        self.camera_frame.grid(
            row = x+1,
            column = y,
            padx = (10, 10),
            pady = (10, 10),
            sticky = "nsew"
        )

        # This is where the camera stream will be seen
        self.canvas = tk.Canvas(
            self.camera_frame, 
            width = self.tk_vid.width, 
            height = self.tk_vid.height
        )
        self.canvas.grid(
            row = x, # y+1, or x+2
            column = y, # x
            padx = 10,
            pady = 10
        )

        # Give equal weight to rows and cols of camera_frame
        super(tkCamera, self).equal_weight(
            self.camera_frame.grid_size(),
            self.camera_frame
        )

        # Frame for buttons
        self.btn_frame = ttk.LabelFrame(
            self.camera_frame,
            text = "Controls",
            padding = (10, 10)
        )
        self.btn_frame.grid(
            row = x+1,
            column = y,
            padx = (10, 10),
            pady = (10, 10)
        )

        # Button that lets the user start camera stream
        self.btn_start = ttk.Button(
            self.btn_frame, 
            text = "Start",
            style = "Accent.TButton", 
            command = self.start
        )
        self.btn_start.grid(
            row = x-x, # x+3
            column = y-y,
            padx = 10,
            pady = 10
        )
        # Button that lets user stop camera stream
        self.btn_pause = ttk.Button(
            self.btn_frame,
            text = "Pause",
            style = "Accent.TButton", 
            command = self.pause
        )
        self.btn_pause.grid(
            row = x-x, # x+3
            column = y-y+1,
            padx = 10,
            pady = 10
        )
        # Button that lets user take a snapshot
        self.btn_snapshot = ttk.Button(
            self.btn_frame,
            text = "Snapshot",
            style = "Accent.TButton", 
            command = self.snapshot
        )
        self.btn_snapshot.grid(
            row = x-x, # x+3
            column = y-y+2,
            padx = 10,
            pady = 10
        )

        self.model_state = tk.IntVar(
            master = self.camera_frame, 
            value = 1
        )
        self.check_box = ttk.Checkbutton(
            self.btn_frame,
            text = "Model",
            variable = self.model_state,
            command = self.change_model_state,
            onvalue = 1,
            offvalue = 0
        )
        self.check_box.grid(
           row = 0,
           column = 3,
           padx = 10,
           pady = 10
        )

        # Give equal weight to rows and cols of btn_frame
        super(tkCamera, self).equal_weight(
            self.btn_frame.grid_size(),
            self.btn_frame
        )

        # Make each tkCamera object responsive to window
        # adjustments. Useful with multiple tkCamera(s).
        super(tkCamera, self).make_responsive(
            self.grid_size(),
            self
        )

        # After it is called once, the update method will be 
        # automatically called every delay in milliseconds.
        # Calculate delay using `FPS`
        self.delay = int(1000/self.tk_vid.fps)

        print('[tkCamera] source:', self.video_source)
        print('[tkCamera] fps:', self.tk_vid.fps, 'delay:', self.delay)
        
        self.image = None
        
        self.running = True
        self.update_frame()

    # Turn the model on or off
    def change_model_state(self):
        if self.model_state.get() == 0:
            self.tk_vid.model_flag = False
        else:
            self.tk_vid.model_flag = True

    def start(self):
        if not self.running:
            self.running = True
            self.update_frame()

    def pause(self):
        if self.running:
           self.running = False
    
    def snapshot(self):
        # Save current frame in canvas
        if self.image:
            self.img_name = time.strftime("frame-%d-%m-%Y-%H-%M-%S.jpg")
            self.img_path = f"{self.folder_path}\{self.img_name}" # /
            self.image.save(self.img_path)
            
    def update_frame(self):
        # Get a frame from the video source
        ret, frame = self.tk_vid.get_frame()
        if ret:
            self.image = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image = self.image)
            self.canvas.create_image(0, 0, image = self.photo, anchor = 'nw')
        
        if self.running:
            self.parent.after(self.delay, self.update_frame)

def main():
    # Driver Code
    app = tkinterApp()
    app.mainloop()

if __name__ == "__main__":
    main()