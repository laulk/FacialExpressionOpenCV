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

        self.btn_undo_smile = tkinter.Button(window, text="Undo Smile/Frown", width = 20, command = self.undo_smile)
        self.btn_undo_smile.pack(anchor = tkinter.CENTER, expand = True)
        self.btn_clearcanvas = tkinter.Button(window, text="Clear Image/Debug", width= 20, command = self.clear_canvas)
        self.btn_clearcanvas.pack(anchor = tkinter.CENTER, expand = True)

        self.btn_saveas = tkinter.Button(window, text = "Save image as", width = 20, command = self.save_fileas)
        self.btn_saveas.pack(anchor = tkinter.CENTER, expand = True)

        self.btn_square = tkinter.Button(window, text="Smile for Square face shape", width = 20, command = self.smile_Square)
        self.btn_square.pack(padx = 5, pady = 10, side = tkinter.LEFT)
        self.btn_round = tkinter.Button(window, text="Smile for Round face shape", width = 20, command = self.smile_Round)
        self.btn_round.pack(padx = 5, pady = 20, side = tkinter.LEFT)
        self.btn_long = tkinter.Button(window, text="Smile for Long face shape", width = 20, command = self.smile_Long)
        self.btn_long.pack(padx = 5, pady = 20, side = tkinter.LEFT)

        

        self.window.mainloop()

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
        

    def smile_Square(self):
        
        points_src = tps.read_landmarks(self.cv_img)
        points_src = np.array(points_src)

        points_dst = points_src + tps.square_face

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


    def smile_Round(self):
        
        points_src = tps.read_landmarks(self.cv_img)
        points_src = np.array(points_src)

        points_dst = points_src + tps.round_face

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

    def smile_Long(self):
        
        points_src = tps.read_landmarks(self.cv_img)
        points_src = np.array(points_src)

        points_dst = points_src + tps.rectangular_face

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
