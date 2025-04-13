from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QFileDialog, QAction
from PyQt5 import uic
from PyQt5.QtGui import QPixmap
import sys
import ysncyhn as yc
import cv2
import numpy as np
import matplotlib.pyplot as plt

class UI(QMainWindow):

    def __init__(self):
        super(UI, self).__init__()

        uic.loadUi("main.ui", self)

        self.buttonResimAc = self.findChild(QAction, "actionResim_A")
        self.buttonPython = self.findChild(QAction, "actionPython")
        self.buttonOpenCv = self.findChild(QAction, "actionOpenCv")
        self.buttonPython_2 = self.findChild(QAction, "actionPython_2")
        self.buttonOpenCv_2 = self.findChild(QAction, "actionOpenCv_2")
        self.buttonPython_3 = self.findChild(QAction, "actionPython_3")
        self.buttonOpenCv_3 = self.findChild(QAction, "actionOpenCv_3")
        self.buttonPython_4 = self.findChild(QAction, "actionPython_4")
        self.buttonOpenCv_4 = self.findChild(QAction, "actionOpenCv_4")
        self.buttonPython_5 = self.findChild(QAction, "actionPython_5")
        self.buttonOpenCv_5 = self.findChild(QAction, "actionOpenCv_5")
        self.buttonPython_6 = self.findChild(QAction, "actionPython_6")
        self.buttonOpenCv_6 = self.findChild(QAction, "actionOpenCv_6")
        self.buttonPython_7 = self.findChild(QAction, "actionPython_7")
        self.buttonOpenCv_7 = self.findChild(QAction, "actionOpenCv_7")
        self.buttonPython_8 = self.findChild(QAction, "actionPython_8")
        self.buttonOpenCv_8 = self.findChild(QAction, "actionOpenCv_8")
        self.buttonPython_9 = self.findChild(QAction, "actionPython_9")
        self.buttonOpenCv_9 = self.findChild(QAction, "actionOpenCv_9")
        self.buttonPython_10 = self.findChild(QAction, "actionPython_10")
        self.buttonOpenCv_10 = self.findChild(QAction, "actionOpenCv_10")
        self.buttonPython_11 = self.findChild(QAction, "actionPython_11")
        self.buttonOpenCv_11 = self.findChild(QAction, "actionOpenCv_11")
        self.buttonPython_12 = self.findChild(QAction, "actionPython_12")
        self.buttonOpenCv_12 = self.findChild(QAction, "actionOpenCv_12")
        self.buttonPython_13 = self.findChild(QAction, "actionPython_13")
        self.buttonOpenCv_13 = self.findChild(QAction, "actionOpenCv_13")
        self.buttonPython_14 = self.findChild(QAction, "actionPython_14")
        self.buttonOpenCv_14 = self.findChild(QAction, "actionOpenCv_14")
        self.buttonPython_15 = self.findChild(QAction, "actionPython_15")
        self.buttonOpenCv_15 = self.findChild(QAction, "actionOpenCv_15")
        self.buttonPython_16 = self.findChild(QAction, "actionPython_16")
        self.buttonOpenCv_16 = self.findChild(QAction, "actionOpenCv_16")
        self.buttonPython_17 = self.findChild(QAction, "actionPython_17")
        self.buttonOpenCv_17 = self.findChild(QAction, "actionOpenCv_17")
        self.buttonPython_18 = self.findChild(QAction, "actionPython_18")
        self.buttonOpenCv_18 = self.findChild(QAction, "actionOpenCv_18")
        self.buttonPython_19 = self.findChild(QAction, "actionPython_19")
        self.buttonOpenCv_19 = self.findChild(QAction, "actionOpenCv_19")
        self.buttonPython_20 = self.findChild(QAction, "actionPython_20")
        self.buttonOpenCv_20 = self.findChild(QAction, "actionOpenCv_20")
        self.buttonPython_21 = self.findChild(QAction, "actionPython_21")
        self.buttonOpenCv_21 = self.findChild(QAction, "actionOpenCv_21")
        self.labelResim = self.findChild(QLabel, "label_2")
        self.labelIslenmisResim = self.findChild(QLabel, "label_4")

        self.buttonResimAc.triggered.connect(self.clicker)
        self.buttonPython.triggered.connect(self.p_rgb_2_gray)
        self.buttonOpenCv.triggered.connect(self.o_rgb_2_gray)
        self.buttonPython_2.triggered.connect(self.p_histogram)
        self.buttonOpenCv_2.triggered.connect(self.o_histogram)
        self.buttonPython_3.triggered.connect(self.p_esikle)
        self.buttonOpenCv_3.triggered.connect(self.o_esikle)
        self.buttonPython_4.triggered.connect(self.p_negatifle)
        self.buttonOpenCv_4.triggered.connect(self.o_negatifle)
        self.buttonPython_5.triggered.connect(self.p_alcak_filtre)
        self.buttonOpenCv_5.triggered.connect(self.o_alcak_filtre)
        self.buttonPython_6.triggered.connect(self.p_gauss_filtre)
        self.buttonOpenCv_6.triggered.connect(self.o_gauss_filtre)
        self.buttonPython_7.triggered.connect(self.p_mean_filtre)
        self.buttonOpenCv_7.triggered.connect(self.o_mean_filtre)
        self.buttonPython_8.triggered.connect(self.p_medyan_filtre)
        self.buttonOpenCv_8.triggered.connect(self.o_medyan_filtre)
        self.buttonPython_9.triggered.connect(self.p_laplac_filtre)
        self.buttonOpenCv_9.triggered.connect(self.o_laplac_filtre)
        self.buttonPython_10.triggered.connect(self.p_sobel_filtre)
        self.buttonOpenCv_10.triggered.connect(self.o_sobel_filtre)
        self.buttonPython_11.triggered.connect(self.p_prewitt_filtre)
        self.buttonOpenCv_11.triggered.connect(self.o_prewitt_filtre)
        self.buttonPython_12.triggered.connect(self.p_kontrast_germe)
        self.buttonOpenCv_12.triggered.connect(self.o_kontrast_germe)
        self.buttonPython_13.triggered.connect(self.p_histogram_esitleme)
        self.buttonOpenCv_13.triggered.connect(self.o_histogram_esitleme)
        self.buttonPython_14.triggered.connect(self.p_yayma)
        self.buttonOpenCv_14.triggered.connect(self.o_yayma)
        self.buttonPython_15.triggered.connect(self.p_asindirma)
        self.buttonOpenCv_15.triggered.connect(self.o_asindirma)
        self.buttonPython_16.triggered.connect(self.p_open)
        self.buttonOpenCv_16.triggered.connect(self.o_open)
        self.buttonPython_17.triggered.connect(self.p_close)
        self.buttonOpenCv_17.triggered.connect(self.o_close)
        self.buttonPython_18.triggered.connect(self.p_netlestir)
        self.buttonOpenCv_18.triggered.connect(self.o_netlestir)
        self.buttonPython_19.triggered.connect(self.p_konvolusyon)
        self.buttonOpenCv_19.triggered.connect(self.o_konvolusyon)
        self.buttonPython_20.triggered.connect(self.p_cevir)
        self.buttonOpenCv_20.triggered.connect(self.o_cevir)
        self.buttonPython_21.triggered.connect(self.p_boyutlandir)
        self.buttonOpenCv_21.triggered.connect(self.o_boyutlandir)

        self.show()
    
    def clicker(self):
        f_name = QFileDialog.getOpenFileName(self, "Open File", "images/", "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg)")
        self.resim_yolu = f_name[0]
        self.pixmap = QPixmap(f_name[0])
        self.labelResim.setPixmap(self.pixmap)

    def p_rgb_2_gray(self):
        resim = yc.f_im_read(self.resim_yolu)
        resim_isle = yc.f_rgb_to_gray(resim)
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)
    def o_rgb_2_gray(self):
        resim = yc.f_im_read(self.resim_yolu)
        resim_isle = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def p_histogram(self):
        resim = yc.f_im_read(self.resim_yolu)
        yc.f_histogram(resim)

    def o_histogram(self):
        resim = yc.f_im_read(self.resim_yolu)
        resim = yc.f_rgb_to_gray(resim)
        plt.hist(resim.ravel(), 256, [0,256])
        plt.show()

    def p_esikle(self):
        resim = yc.f_im_read(self.resim_yolu)
        resim_isle = yc.f_esikle(resim)
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_esikle(self):
        print("o")

    def p_negatifle(self):
        resim = yc.f_im_read(self.resim_yolu)
        resim_isle = yc.f_negatifle(resim)
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_negatifle(self):
        print("o")

    def p_alcak_filtre(self):
        resim = yc.f_im_read(self.resim_yolu)
        maske = yc.f_matris_one(4)
        resim_isle = yc.f_maske(resim, maske)
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_alcak_filtre(self):
        resim = yc.f_rgb_to_gray(yc.f_im_read(self.resim_yolu))
        resim_isle = cv2.blur(resim, (5, 5))
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def p_gauss_filtre(self):
        resim = yc.f_im_read(self.resim_yolu)
        maske = yc.f_fspecial("gaussian")
        resim_isle = yc.f_maske(resim, maske)
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_gauss_filtre(self):
        resim = yc.f_rgb_to_gray(yc.f_im_read(self.resim_yolu))
        resim_isle = cv2.GaussianBlur ( resim , ( 5 , 5 ) , cv2.BORDER_DEFAULT )
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)


    def p_mean_filtre(self):
        resim = yc.f_im_read(self.resim_yolu)
        maske = yc.f_matris_one(7)
        resim_isle = yc.f_maske(resim, maske)
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_mean_filtre(self):
        print("o")

    def p_medyan_filtre(self):
        resim = yc.f_im_read(self.resim_yolu)
        resim_isle = yc.f_medyan(resim)
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_medyan_filtre(self):
        resim = yc.f_im_read(self.resim_yolu)
        resim_isle = cv2.medianBlur(resim, ksize=7)
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def p_laplac_filtre(self):
        resim = yc.f_im_read(self.resim_yolu)
        maske = yc.f_fspecial("laplacian")
        resim_filtre = yc.f_maske(resim, maske)
        gri_resim = yc.f_rgb_to_gray(resim)
        resim_isle = gri_resim - resim_filtre
        yc.f_im_write(resim_isle)
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_laplac_filtre(self):
        resim = yc.f_im_read(self.resim_yolu)
        resim_isle = cv2.Laplacian(resim, cv2.CV_64F)
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def p_sobel_filtre(self):
        resim = yc.f_im_read(self.resim_yolu)
        yatay_maske = yc.f_fspecial("sobel_yatay")
        dikey_maske = yc.f_fspecial("sobel_dikey")
        sobel_horizontal = yc.f_maske(resim, yatay_maske)
        sobel_vertical = yc.f_maske(resim, dikey_maske)
        yc.f_im_write(sobel_horizontal)
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_sobel_filtre(self):
        resim = yc.f_im_read(self.resim_yolu)
        sobel_horizontal = cv2.Sobel(resim, cv2.CV_64F, 1, 0, ksize=5)
        sobel_vertical = cv2.Sobel(resim, cv2.CV_64F, 0, 1, ksize=5)
        yc.f_im_write(sobel_horizontal)
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def p_prewitt_filtre(self):
        resim = yc.f_im_read(self.resim_yolu)
        yatay_maske = yc.f_fspecial("prewitt_yatay")
        dikey_maske = yc.f_fspecial("prewitt_dikey")
        prewitt_horizontal = yc.f_maske(resim, yatay_maske)
        prewitt_vertical = yc.f_maske(resim, dikey_maske)
        yc.f_im_write(prewitt_horizontal)
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_prewitt_filtre(self):
        print("o")

    def p_kontrast_germe(self):
        resim = yc.f_im_read(self.resim_yolu)
        resim_isle = yc.f_kontrast_germe(resim)
        yc.f_im_write(resim_isle)
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)
        yc.f_histogram(resim_isle)

    def o_kontrast_germe(self):
        print("o")

    def p_histogram_esitleme(self):
        print("p")

    def o_histogram_esitleme(self):
        print("o")

    def p_yayma(self):
        resim = yc.f_im_read(self.resim_yolu)
        sb_resim = yc.f_esikle(resim)
        resim_isle = yc.f_yayma(sb_resim)
        yc.f_im_write(resim_isle)
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_yayma(self):
        resim = yc.f_im_read(self.resim_yolu)
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        resim_isle = cv2.dilate(resim, kernel, iterations = 1)
        yc.f_im_write(resim_isle)
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def p_asindirma(self):
        resim = yc.f_im_read(self.resim_yolu)
        sb_resim = yc.f_esikle(resim)
        resim_isle = yc.f_asindirma(sb_resim)
        yc.f_im_write(resim_isle)
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_asindirma(self):
        resim = yc.f_im_read(self.resim_yolu)
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        resim_isle = cv2.erode(resim, kernel, iterations = 1)
        yc.f_im_write(resim_isle)
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)
    
    def p_open(self):
        resim = yc.f_im_read(self.resim_yolu)
        sb_resim = yc.f_esikle(resim)
        resim_isle = yc.f_asindirma(sb_resim)
        resim_isle = yc.f_yayma(resim_isle)
        yc.f_im_write(resim_isle)
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_open(self):
        resim = yc.f_im_read(self.resim_yolu)
        kernel = np.ones((3, 3), np.uint8)
        resim_isle = cv2.morphologyEx(resim, cv2.MORPH_OPEN, kernel, iterations=1) 
        yc.f_im_write(resim_isle)
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)
    
    def p_close(self):
        resim = yc.f_im_read(self.resim_yolu)
        sb_resim = yc.f_esikle(resim)
        resim_isle = yc.f_yayma(sb_resim)
        resim_isle = yc.f_asindirma(resim_isle)
        yc.f_im_write(resim_isle)
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_close(self):
        resim = yc.f_im_read(self.resim_yolu)
        kernel = np.ones((3, 3), np.uint8)
        resim_isle = cv2.morphologyEx(resim, cv2.MORPH_CLOSE, kernel, iterations=1) 
        yc.f_im_write(resim_isle)
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def p_netlestir(self):
        resim = yc.f_im_read(self.resim_yolu)
        maske = yc.f_fspecial("netles")
        resim_isle = yc.f_maske(resim, maske)
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_netlestir(self):
        print("o")

    def p_konvolusyon(self):
        resim = yc.f_im_read(self.resim_yolu)
        maske = yc.f_fspecial("konvolusyon")
        resim_isle = yc.f_konvolusyon(resim, maske)
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def o_konvolusyon(self):
        print("o")


    def p_cevir(self):
        print("p") 

    def o_cevir(self):
        resim = yc.f_im_read(self.resim_yolu)
        height, width = resim.shape[:2]   #Görüntüyü merkezin etrafında döndürmek için ikiye bölün.
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, .5)
        rotated_image = cv2.warpAffine(resim, rotation_matrix, (width, height))
        yc.f_im_write(rotated_image) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)

    def p_boyutlandir(self):
        print("p") 

    def o_boyutlandir(self):
        resim = yc.f_im_read(self.resim_yolu)
        resim_isle = cv2.resize(resim, (300, 300))
        yc.f_im_write(resim_isle) 
        self.pixmapIsle = QPixmap('images\\resim.png')
        self.labelIslenmisResim.setPixmap(self.pixmapIsle)




app = QApplication(sys.argv)
UIWindow = UI()
app.exec_()