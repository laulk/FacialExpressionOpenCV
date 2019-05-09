import tps
import tkinter
import cv2
import PIL.Image, PIL.ImageTk, PIL.ImageGrab
from tkinter import filedialog
import numpy as np
import sys
import dlib
import functools
import imutils
from imutils import face_utils
import math

class App:
    def __init__(self, window, window_title):
        self.canvasb4 = tkinter.Canvas(window, width = 250, height = 300)
        self.canvasb4.pack(padx = 10, pady = 20, side = tkinter.LEFT)
        self.canvas = tkinter.Canvas(window, width = 250, height = 300)
        self.canvas.pack(padx = 10, pady = 20, side = tkinter.LEFT)
        self.window = window
        self.window.title(window_title)
        self.btn_selectimg = tkinter.Button(window, text = "Browse for Image", width =20, command = self.select_image)
        self.btn_selectimg.pack(anchor = tkinter.CENTER, expand = True)
        self.btn_smile = tkinter.Button(window, text="Make Face Smile", width = 20, command = self.smile_image)
        self.btn_smile.pack(anchor = tkinter.CENTER, expand=True)

        self.scaler_smile = tkinter.Scale(window, label = "Degree of smile",orient = 'horizontal', length = 300, width=20, from_= 0 , to=10, tickinterval = 1, command = self.Slider_smile)
        self.scaler_smile.pack(anchor = tkinter.CENTER, expand = True)
    
        self.btn_undo_smile = tkinter.Button(window, text="Undo Smile/Frown", width = 20, command = self.undo_smile)
        self.btn_undo_smile.pack(anchor = tkinter.CENTER, expand = True)
        self.btn_clearcanvas = tkinter.Button(window, text="Clear Image/Debug", width= 20, command = self.clear_canvas)
        self.btn_clearcanvas.pack(anchor = tkinter.CENTER, expand = True)

        self.btn_saveas = tkinter.Button(window, text = "Save image as", width = 20, command = self.save_fileas)
        self.btn_saveas.pack(anchor = tkinter.CENTER, expand = True)

        self.window.mainloop()

    def Slider_smile(self, val):

        #print(type(val))
        points_src = tps.read_landmarks(self.cv_img)
        points_src = np.array(points_src)

        if int(val) > 0 and int(val) < 2:
                
            theval = int(val)
            #tps.small_offset_increment = tps.small_offset_increment * theval
            points_dst = points_src + tps.small_offset_increment

            tshape = np.array(points_dst, np.float32)
            sshape = np.array(points_src, np.float32)

            spline =  cv2.createThinPlateSplineShapeTransformer()

            sshape = sshape.reshape(1,-1,2)
            tshape = tshape.reshape(1,-1,2)

            matches = list()
            n = 0
            while n <72:
                matches.append(cv2.DMatch(n,n,n))
                n += 1
            
            spline.estimateTransformation(tshape, sshape, matches)

            self.cv_deform = self.cv_img
            self.cv_deform = spline.warpImage(self.cv_deform)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_deform))
            self.canvas.create_image(0,0,image = self.photo, anchor = tkinter.NW)
        
        elif int(val) == 2:
            points_dst = points_src + tps.small_offset_increment

            tshape = np.array(points_dst, np.float32)
            sshape = np.array(points_src, np.float32)

            spline =  cv2.createThinPlateSplineShapeTransformer()

            sshape = sshape.reshape(1,-1,2)
            tshape = tshape.reshape(1,-1,2)

            matches = list()
            n = 0
            while n <72:
                matches.append(cv2.DMatch(n,n,n))
                n += 1
            
            spline.estimateTransformation(tshape, sshape, matches)

            #self.cv_deform = self.cv_img
            self.cv_deform = spline.warpImage(self.cv_deform)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_deform))
            self.canvas.create_image(0,0,image = self.photo, anchor = tkinter.NW)

        elif int(val) == 3:
            points_dst = points_src + tps.small_offset_increment

            tshape = np.array(points_dst, np.float32)
            sshape = np.array(points_src, np.float32)

            spline =  cv2.createThinPlateSplineShapeTransformer()

            sshape = sshape.reshape(1,-1,2)
            tshape = tshape.reshape(1,-1,2)

            matches = list()
            n = 0
            while n <72:
                matches.append(cv2.DMatch(n,n,n))
                n += 1
            
            spline.estimateTransformation(tshape, sshape, matches)

            #self.cv_deform = self.cv_img
            self.cv_deform = spline.warpImage(self.cv_deform)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_deform))
            self.canvas.create_image(0,0,image = self.photo, anchor = tkinter.NW)
        elif int(val) == 4:
            points_dst = points_src + tps.small_offset_increment

            tshape = np.array(points_dst, np.float32)
            sshape = np.array(points_src, np.float32)

            spline =  cv2.createThinPlateSplineShapeTransformer()

            sshape = sshape.reshape(1,-1,2)
            tshape = tshape.reshape(1,-1,2)

            matches = list()
            n = 0
            while n <72:
                matches.append(cv2.DMatch(n,n,n))
                n += 1
            
            spline.estimateTransformation(tshape, sshape, matches)

            #self.cv_deform = self.cv_img
            self.cv_deform = spline.warpImage(self.cv_deform)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_deform))
            self.canvas.create_image(0,0,image = self.photo, anchor = tkinter.NW)

        elif int(val) == 5:
            points_dst = points_src + tps.small_offset_increment

            tshape = np.array(points_dst, np.float32)
            sshape = np.array(points_src, np.float32)

            spline =  cv2.createThinPlateSplineShapeTransformer()

            sshape = sshape.reshape(1,-1,2)
            tshape = tshape.reshape(1,-1,2)

            matches = list()
            n = 0
            while n <72:
                matches.append(cv2.DMatch(n,n,n))
                n += 1
            
            spline.estimateTransformation(tshape, sshape, matches)

            #self.cv_deform = self.cv_img
            self.cv_deform = spline.warpImage(self.cv_deform)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_deform))
            self.canvas.create_image(0,0,image = self.photo, anchor = tkinter.NW)

        elif int(val) == 6:
            points_dst = points_src + tps.small_offset_increment

            tshape = np.array(points_dst, np.float32)
            sshape = np.array(points_src, np.float32)

            spline =  cv2.createThinPlateSplineShapeTransformer()

            sshape = sshape.reshape(1,-1,2)
            tshape = tshape.reshape(1,-1,2)

            matches = list()
            n = 0
            while n <72:
                matches.append(cv2.DMatch(n,n,n))
                n += 1
            
            spline.estimateTransformation(tshape, sshape, matches)

            #self.cv_deform = self.cv_img
            self.cv_deform = spline.warpImage(self.cv_deform)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_deform))
            self.canvas.create_image(0,0,image = self.photo, anchor = tkinter.NW)            

        elif int(val) == 7:
            points_dst = points_src + tps.small_offset_increment

            tshape = np.array(points_dst, np.float32)
            sshape = np.array(points_src, np.float32)

            spline =  cv2.createThinPlateSplineShapeTransformer()

            sshape = sshape.reshape(1,-1,2)
            tshape = tshape.reshape(1,-1,2)

            matches = list()
            n = 0
            while n <72:
                matches.append(cv2.DMatch(n,n,n))
                n += 1
            
            spline.estimateTransformation(tshape, sshape, matches)

            #self.cv_deform = self.cv_img
            self.cv_deform = spline.warpImage(self.cv_deform)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_deform))
            self.canvas.create_image(0,0,image = self.photo, anchor = tkinter.NW)        

        elif int(val) == 8:
            points_dst = points_src + tps.small_offset_increment

            tshape = np.array(points_dst, np.float32)
            sshape = np.array(points_src, np.float32)

            spline =  cv2.createThinPlateSplineShapeTransformer()

            sshape = sshape.reshape(1,-1,2)
            tshape = tshape.reshape(1,-1,2)

            matches = list()
            n = 0
            while n <72:
                matches.append(cv2.DMatch(n,n,n))
                n += 1
            
            spline.estimateTransformation(tshape, sshape, matches)

            #self.cv_deform = self.cv_img
            self.cv_deform = spline.warpImage(self.cv_deform)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_deform))
            self.canvas.create_image(0,0,image = self.photo, anchor = tkinter.NW)

        elif int(val) == 9:
            points_dst = points_src + tps.small_offset_increment

            tshape = np.array(points_dst, np.float32)
            sshape = np.array(points_src, np.float32)

            spline =  cv2.createThinPlateSplineShapeTransformer()

            sshape = sshape.reshape(1,-1,2)
            tshape = tshape.reshape(1,-1,2)

            matches = list()
            n = 0
            while n <72:
                matches.append(cv2.DMatch(n,n,n))
                n += 1
            
            spline.estimateTransformation(tshape, sshape, matches)

            #self.cv_deform = self.cv_img
            self.cv_deform = spline.warpImage(self.cv_deform)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_deform))
            self.canvas.create_image(0,0,image = self.photo, anchor = tkinter.NW)

        elif int(val) == 10:
            points_dst = points_src + tps.small_offset_increment

            tshape = np.array(points_dst, np.float32)
            sshape = np.array(points_src, np.float32)

            spline =  cv2.createThinPlateSplineShapeTransformer()

            sshape = sshape.reshape(1,-1,2)
            tshape = tshape.reshape(1,-1,2)

            matches = list()
            n = 0
            while n <72:
                matches.append(cv2.DMatch(n,n,n))
                n += 1
            
            spline.estimateTransformation(tshape, sshape, matches)

            #self.cv_deform = self.cv_img
            self.cv_deform = spline.warpImage(self.cv_deform)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_deform))
            self.canvas.create_image(0,0,image = self.photo, anchor = tkinter.NW)

    def select_image(self):
        image_path = filedialog.askopenfilename()
        
        self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        self.height, self.width, no_channels = self.cv_img.shape
        
        #self.canvas = tkinter.Canvas(width = self.width, height = self.height)
        #self.canvas.pack()
        self.photob4 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvasb4.create_image(0, 0, image = self.photob4, anchor = tkinter.NW)
        self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

    def save_fileas(self):
        x = self.canvas.winfo_rootx() + self.canvas.winfo_x()
        #'''self.winfo_rootx() + ''' 
        y = self.canvas.winfo_rooty() + self.canvas.winfo_y()
        #'''self.winfo_rooty() + ''' 
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        filename = filedialog.asksaveasfilename(title = "Select file",filetypes = (("PNG files","*.png"),("All files", "*.*")),defaultextension = ".png")
        PIL.ImageGrab.grab().crop((x,y,x1,y1)).save(filename)

    def smile_image(self):
        
        points_src = tps.read_landmarks(self.cv_img)
        points_src = np.array(points_src)

        points_dst = points_src + tps.Smile_offset

        tshape = np.array(points_dst, np.float32)
        sshape = np.array(points_src, np.float32)

        spline =  cv2.createThinPlateSplineShapeTransformer()

        sshape = sshape.reshape(1,-1,2)
        tshape = tshape.reshape(1,-1,2)

        matches = list()
        n = 0
        while n <72:
            matches.append(cv2.DMatch(n,n,n))
            n += 1
        
        spline.estimateTransformation(tshape, sshape, matches)

        self.cv_img = spline.warpImage(self.cv_img)
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0,0,image = self.photo, anchor = tkinter.NW)

    def undo_smile(self):
        points_src = tps.read_landmarks(self.cv_img)
        points_src = np.array(points_src)

        points_dst = points_src - tps.Smile_offset

        tshape = np.array(points_dst, np.float32)
        sshape = np.array(points_src, np.float32)

        spline =  cv2.createThinPlateSplineShapeTransformer()

        sshape = sshape.reshape(1,-1,2)
        tshape = tshape.reshape(1,-1,2)

        matches = list()
        n = 0
        while n <72:
            matches.append(cv2.DMatch(n,n,n))
            n += 1
        
        spline.estimateTransformation(tshape, sshape, matches)

        self.cv_img = spline.warpImage(self.cv_img)
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0,0,image = self.photo, anchor = tkinter.NW)
        

    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvasb4.delete("all")
        

App(tkinter.Tk(), "Expression Transformer")
