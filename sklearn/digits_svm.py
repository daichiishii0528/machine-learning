
# coding: utf-8

# In[ ]:

# データセットを取得
from sklearn import datasets
digits = datasets.load_digits()

# 画像にして出力
import matplotlib.pyplot as plt
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(2, 5, index + 1)
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.axis('off')
    plt.title('Training: %i' % label)
plt.show()


# In[ ]:

# SVMをインポート
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)

# train SVM
clf.fit(digits.data[:n_samples * 6 / 10 ], digits.target[:n_samples * 6 / 10])


# In[ ]:

# 可視化
print(digits.target[-10:]) # 正解ラベル
print(clf.predict(digits.data[-10:])) # 予測ラベル


# In[ ]:

# 正答率，f値の出力
expected = digits.target[n_samples * -4 / 10:] # 正解ラベル
predicted = clf.predict(digits.data[n_samples * -4 / 10:]) # 予測ラベル
from sklearn import metrics
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))


# In[ ]:

# 元データと予測結果を一覧出力
images_and_predictions = list(zip(digits.images[n_samples * -4 / 10:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:12]):
    plt.subplot(3, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)
plt.show()         

