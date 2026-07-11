from tabpfn import TabPFNClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = TabPFNClassifier()
clf.fit(X_train, y_train)   # this is NOT training — just caches the context
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)