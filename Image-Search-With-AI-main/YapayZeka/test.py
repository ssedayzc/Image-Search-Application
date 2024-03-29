import cv2
import numpy as np
from scipy.cluster.vq import *
from sklearn.preprocessing import normalize
import os
from PIL import Image

# SIFT nesnesini oluşturun
sift = cv2.SIFT_create(1000)

# Kaydedilmiş merkezleri, görüntü özelliklerini ve görüntü listesini yükleyin
centers = np.load("centers.npy")
img_feats = np.load("imgfeats.npy")
img_list = np.load("imgmap.npy")

# K değerini tanımlayın
K = 256

def test(im):
    # Giriş görüntüsünden SIFT özelliklerini çıkarın
    _ , des = sift.detectAndCompute(im, None)

    # Test özellik vektörünü oluşturun
    test_feats = np.zeros(K)
    
    # En yakın merkezlere göre vektörü doldurun
    closest, _ = vq(des, centers)
    np.add.at(test_feats, closest, 1)
    
    # Normalizasyon işlemi
    test_feats = normalize(test_feats.reshape(1,-1))[0]

    # En iyi beş eşleşmeyi bulun
    TOP_FIVE = [img_list[x] for x in np.argsort(np.dot(test_feats, img_feats.T)).tolist()[::-1][:5]]

    # En iyi beş görüntüyü yükleyin ve RGB renk formatına dönüştürün
    result_images = [cv2.cvtColor(cv2.imread(f"database/{i}"), cv2.COLOR_BGR2RGB) for i in TOP_FIVE]
    
    return result_images

# Test için kullanılacak bir örnek görüntüyü yükleyin
sample_image_path = "var2.jpg"

sample_image = cv2.cvtColor(cv2.imread(sample_image_path), cv2.COLOR_BGR2RGB)

# Test işlemini gerçekleştirin
result_images = test(sample_image)

# Test sonuçlarını gösterin
for i, result_image in enumerate(result_images):
    Image.fromarray(result_image).show()
