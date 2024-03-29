# 🖼️ Görüntü İşleme ve Eşleştirme

Bu proje, OpenCV ve Python kullanarak görüntü işleme ve eşleştirme yöntemlerini uygulamak için bir örnektir. Kullanıcı, veritabanında bulunan görüntülerin SIFT (Scale-Invariant Feature Transform) özelliklerini çıkarabilir ve ardından yeni bir görüntüyü bu veritabanındaki görüntülerle karşılaştırabilir.

## 📋 Gereksinimler

Bu projeyi çalıştırmak için şu yazılım ve kütüphanelere ihtiyacınız vardır:

- Python 3.x
- OpenCV
- NumPy
- scikit-learn
- PIL (Python Imaging Library)

Yukarıdaki kütüphaneleri yüklemek için terminal veya komut istemcisinde şu komutları kullanabilirsiniz:

```bash
pip install opencv-python numpy scikit-learn pillow

▶️ Nasıl Kullanılır
1- database klasörüne, veritabanında karşılaştırmak istediğiniz görüntüleri ekleyin.
2- train.py dosyasını çalıştırarak, veritabanındaki görüntülerin SIFT özelliklerini çıkarın ve K-means kümeleme algoritması kullanarak görüntüleri gruplayın.
3- Ardından test.py dosyasını çalıştırarak, yeni bir görüntüyü karşılaştırın ve en benzer görüntüleri bulun.

Örnek olarak:
python train.py
python test.py test/yeni_goruntu.jpg

