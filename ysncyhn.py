import cv2
import numpy as np
import matplotlib.pyplot as plt

def f_im_read(resim_yolu):
    resim = cv2.imread(resim_yolu)
    return resim

def f_im_show(baslik, resim):
    cv2.imshow(baslik, resim)

def f_im_write(resim):
    cv2.imwrite('images\\resim.png', resim) 

    

def f_matris_none(*boyut):

    try:
        matris =  np.zeros((boyut[0], boyut[1], boyut[2]), dtype=np.uint8) # tamamı 0 matris oluşturma
    except:
        matris =  np.zeros((boyut[0], boyut[1]), dtype=np.uint8) # tamamı 0 matris oluşturma

    return matris

def f_matris_one(boyut):
    maske = np.ones((boyut, boyut)) # m * n lük birim matris oluşturma
    return maske

def f_liste(resim, yukseklik, genislik):
    liste = []

    for h in range(yukseklik):
        for w in range(genislik):
            liste.append(resim[h][w])

    return liste

def f_aralik(min, max):
    aralik =  []
    max += 1

    for a in range(min, max):
        aralik.append(int(a))

    return aralik

def f_m_tp(maske):
    m_boyut_y, m_boyut_x  = maske.shape
    m_boyut = m_boyut_x
    tpbln = 0
    for i in range(m_boyut):
        for j in range(m_boyut):
            tpbln += maske[i][j]
    if int(tpbln) == 0:
        tpbln = 1

    return int(tpbln)

def f_fspecial(filtre):
    if filtre == "gaussian":
        maske = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])
    if filtre == "lapla":
        maske = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    if filtre == "laplacian":
        maske = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    if filtre == "sobel_yatay":
        maske = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    if filtre == "sobel_dikey":
        maske = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    if filtre == "prewitt_yatay":
        maske = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    if filtre == "prewitt_dikey":
        maske = np.array([[-1, -0, -1], [-1, 0, 1], [-1, 0, 1]])
    if filtre == "netles":
        maske = np.array([[0, -2, 0], [-2, 11, -2], [0, -2, 0]])
    if filtre == "konvolusyon":
        maske = np.array([[-1, -1, -1], [-2, 10, -2], [-1, -1, -1]])
    return maske

def f_esik_degeri(g_resim):
    try:
        yukseklik, genislik, kanal = g_resim.shape
        resim = f_rgb_to_gray(g_resim)
    except:
        yukseklik, genislik = g_resim.shape

    tp = 0

    for h in range(yukseklik):
        for w in range(genislik):
            tp += int(g_resim[h][w])

    p_sayi = yukseklik * genislik

    ort = int(tp / p_sayi)

    return ort

def f_rgb_to_gray(resim):

    yukseklik, genislik, kanal = resim.shape
    # print("Yükseklik: ", yükseklik)
    # print("Genişlik:" , genislik)
    # print("Kanal sayısı: ", kanal)

    g_resim = f_matris_none(yukseklik, genislik)

    # gri resme dönüştürme

    for h in range(yukseklik):
        for w in range(genislik):
            g_resim[h][w] = int((resim[h][w][0] * 0.299) + (resim[h][w][1] * 0.587) + (resim[h][w][2] * 0.114))

    return g_resim

def f_histogram(resim):
    try:
        yukseklik, genislik, kanal = resim.shape
        resim = f_rgb_to_gray(resim)
    except:
        yukseklik, genislik = resim.shape

    # histagram görüntüleme

    piksel_degerleri =  f_liste(resim, yukseklik, genislik)

    aralik = f_aralik(0, 255)

    plt.hist(piksel_degerleri, aralik)
    plt.show() # histagram görüntüleme

def f_esikle(resim):
    try:
        yukseklik, genislik, kanal = resim.shape
        resim = f_rgb_to_gray(resim)
    except:
        yukseklik, genislik = resim.shape

    s_b_resim = f_matris_none(yukseklik, genislik)
    
    # eşikleme

    e_degeri = f_esik_degeri(resim)

    for h in range(yukseklik):
        for w in range(genislik):
            if resim[h][w] >= e_degeri:
                s_b_resim[h][w] = 255
            else:
                s_b_resim[h][w] = 0

    return s_b_resim

def f_negatifle(resim):
    try:
        yukseklik, genislik, kanal = resim.shape
        resim = f_rgb_to_gray(resim)
    except:
        yukseklik, genislik = resim.shape

    n_resim = f_matris_none(yukseklik, genislik)
    
    # negatifleme

    for h in range(yukseklik):
        for w in range(genislik):
            n_resim[h][w] = 255 - resim[h][w]

    return n_resim

def f_maske(resim, maske):
    try:
        yukseklik, genislik, kanal = resim.shape
        resim = f_rgb_to_gray(resim)
    except:
        yukseklik, genislik = resim.shape

    m_boyut_y, m_boyut_x  = maske.shape
    m_boyut = m_boyut_x
    bolen = f_m_tp(maske)

    filtre_resim = f_matris_none(yukseklik, genislik)

    if m_boyut%2==1:
        a = int((m_boyut-1) / 2)

        
    if m_boyut%2 == 0:
        for h in range(yukseklik):
            for w in range(genislik):
                yeniDeger = 0
                for i in range(m_boyut):
                    for j in range(m_boyut):
                        try:
                            yeniDeger += int(maske[i][j] / bolen * resim[h + i][w + j])
                        except:
                            yeniDeger += 0
                filtre_resim[h][w] = yeniDeger
    else:
        for h in range(yukseklik):
            for w in range(genislik):
                yeniDeger = 0
                for i in range(m_boyut):
                    for j in range(m_boyut):
                        try:
                            yeniDeger += int(maske[i][j] / bolen * resim[h + i - a][w + j - a])
                        except:
                            yeniDeger += 0
                filtre_resim[h][w] = yeniDeger

    return filtre_resim

def f_medyan(resim):
    try:
        yukseklik, genislik, kanal = resim.shape
        resim = f_rgb_to_gray(resim)
    except:
        yukseklik, genislik = resim.shape
    filtre_resim = f_matris_none(yukseklik, genislik)

    m_boyut = 3
    a = int((m_boyut-1) / 2)

    dizi =  []

    for h in range(0 + a, yukseklik - a):
        for w in range(0 + a, genislik - a):
            yeniDeger = 0
            dizi.clear()
            for i in range(0, m_boyut):
                for j in range(0, m_boyut):
                    x = h + i - a
                    y = w + j - a
                    dizi.append(int(resim[x][y]))
            for i in range(9):
                for j in range(i):
                    if dizi[i] < dizi[j]:
                        tut = dizi[j]
                        dizi[j] = dizi[i]
                        dizi[i] = tut
                    else:
                        continue
            filtre_resim[h][w] = dizi[5]

    return filtre_resim


def f_kontrast_germe(resim):
    try:
        yukseklik, genislik, kanal = resim.shape
        resim = f_rgb_to_gray(resim)
    except:
        yukseklik, genislik = resim.shape

    filtre_resim = f_matris_none(yukseklik, genislik)
    piksel_degerleri =  f_liste(resim, yukseklik, genislik)

    eski_en_kucuk = min(piksel_degerleri)
    eski_en_buyuk = max(piksel_degerleri)
    n = 0
    N = 255
    for h in range(yukseklik):
        for w in range(genislik):
            filtre_resim[h][w] = ((((resim[h][w] - eski_en_kucuk) * (N - n)) / (eski_en_buyuk - eski_en_kucuk)) + n)

    return filtre_resim

def f_histogram_esitleme(resim):
    try:
        yukseklik, genislik, kanal = resim.shape
        resim = f_rgb_to_gray(resim)
    except:
        yukseklik, genislik = resim.shape

    filtre_resim = f_matris_none(yukseklik, genislik)
    piksel_degerleri =  f_liste(resim, yukseklik, genislik)

    eski_en_kucuk = min(piksel_degerleri)
    eski_en_buyuk = max(piksel_degerleri)
    n = 0
    N = 255
    for h in range(yukseklik):
        for w in range(genislik):
            filtre_resim[h][w] = ((((resim[h][w] - eski_en_buyuk) * (N - n)) / (eski_en_buyuk - eski_en_kucuk)) + n)

    return filtre_resim

def f_yayma(resim):
    try:
        yukseklik, genislik, kanal = resim.shape
        resim = f_esikle(resim)
    except:
        yukseklik, genislik = resim.shape
    filtre_resim = f_matris_none(yukseklik, genislik)

    m_boyut = 3
    a = int((m_boyut-1) / 2)

    for h in range(0 + a, yukseklik - a):
        for w in range(0 + a, genislik - a):
            yeniDeger = False
            for i in range(m_boyut):
                for j in range(m_boyut):
                    if resim[h + i - a][w + j - a] == 255:
                        yeniDeger = True
                        break
                        break
            if yeniDeger:
                filtre_resim[h][w] = 255
            else:
                filtre_resim[h][w] = 0


    return filtre_resim

def f_asindirma(resim):
    try:
        yukseklik, genislik, kanal = resim.shape
        resim = f_esikle(resim)
    except:
        yukseklik, genislik = resim.shape
    filtre_resim = f_matris_none(yukseklik, genislik)

    m_boyut = 3
    a = int((m_boyut-1) / 2)

    for h in range(0 + a, yukseklik - a):
        for w in range(0 + a, genislik - a):
            yeniDeger = True
            for i in range(m_boyut):
                for j in range(m_boyut):
                    if resim[h + i - a][w + j - a] == 0:
                        yeniDeger = False
                        break
                        break
            if yeniDeger:
                filtre_resim[h][w] = 255
            else:
                filtre_resim[h][w] = 0


    return filtre_resim

def f_konvolusyon(resim, maske):
    try:
        yukseklik, genislik, kanal = resim.shape
        resim = f_rgb_to_gray(resim)
    except:
        yukseklik, genislik = resim.shape

    m_boyut_y, m_boyut_x  = maske.shape
    m_boyut = m_boyut_x
    bolen = f_m_tp(maske)

    filtre_resim = f_matris_none(yukseklik, genislik)

    if m_boyut%2==1:
        a = int((m_boyut-1) / 2)

        

    for h in range(yukseklik):
        for w in range(genislik):
            yeniDeger = 0
            t=0
            l = 0
            for i in range(m_boyut - 1, -1, -1):
                for j in range(m_boyut - 1, -1, -1):
                    try:
                        yeniDeger += int(maske[i][j] / bolen * resim[h + t - a][w + l - a])
                    except:
                        yeniDeger += 0
                    l+=1
                t+=1
            filtre_resim[h][w] = yeniDeger

    return filtre_resim