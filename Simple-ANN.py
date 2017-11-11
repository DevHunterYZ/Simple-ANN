from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
#random_state: rastgele sayı üreteci, solver=Ağırlık optimizasyonu için çözücü, learning_rate_init: ilk öğrenme oranı ,max_iter= maksimum iterasyon(yineleme) sayısı
#hidden_layer_sizes: gizli katmandaki nöron sayıları
mlp = MLPClassifier(hidden_layer_sizes=(50, 4), max_iter=500, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)
mlp.fit(learnset, learnlabels)
print("Training set score: %f" % mlp.score(learnset, learnlabels))
print("Test set score: %f" % mlp.score(learnset, learnlabels))
mlp.classes_
