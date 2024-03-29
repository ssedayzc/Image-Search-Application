import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import normalize
import os

K = 256

feat_list = []
img_list = []
sift = cv2.SIFT_create(nfeatures=1000)

print("SIFT Özellikleri Çıkartılıyor...")

impath = os.listdir("database/")

# Eğitim
for i in impath:
    Path = "database/" + i
    img = cv2.imread(Path)
    colored_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # SIFT özelliklerini çıkart
    _, des = sift.detectAndCompute(colored_img, None)
    
    # SIFT özellikleri boş değilse feat_list'e ekle
    if des is not None and len(des) > 0:
        feat_list.append(des)
        img_list.append(i)

# Hata ayıklama: feat_list içindeki dizilerin boyutlarını kontrol et
for idx, feat in enumerate(feat_list):
    print(f"Index {idx} için dizi boyutu: {feat.shape}")

print("Özellik Çıkartma Tamamlandı")
print("K Means Başlıyor...")

# k-means kümeleme işleminden sonra hesaplanan merkezleri kaydet
centers, _ = kmeans(np.vstack(feat_list), K, 5)
np.save("centers.npy", centers)

print("K Means Tamamlandı")
img_feats = np.zeros((len(feat_list), K), "float32")
for i in range(len(feat_list)):
    closest, _ = vq(feat_list[i], centers)
    np.add.at(img_feats[i], closest, 1)

img_feats = normalize(img_feats)

# Hesaplanan img_feats'i bir dosyaya kaydet
np.save("imgfeats.npy", img_feats)

# img_list'i bir dosyaya kaydet
np.save("imgmap.npy", img_list)

print("Eğitim Tamamlandı...")
