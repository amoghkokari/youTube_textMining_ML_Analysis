from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def mlOps(X_train, X_test, y_train, y_test, model, name):
    dct = {"m_name":name}
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    dct["accuracy"] = round(accuracy_score(y_test,y_pred)*100,2)
    weighted_prf = precision_recall_fscore_support(y_test,y_pred,average='weighted')
    dct["precision"] = round(weighted_prf[0]*100, 2)
    dct["recall"] = round(weighted_prf[1]*100, 2)
    dct["f1score"] = round(weighted_prf[2]*100, 2)
    return dct, model

def clf_main(X_train, X_test, y_train, y_test):

    name = "GaussianNB"
    model = GaussianNB()
    NBdct, NBmodel = mlOps(X_train, X_test, y_train, y_test, model, name)
    print(name+" done")

    name = "LGBMClassifier"
    model = LGBMClassifier()
    LGBMdct, LGBMmodel = mlOps(X_train, X_test, y_train, y_test, model, name)
    print(name+" done")

    name = "XGBClassifier"
    model = XGBClassifier()
    XGBdct, XGBmodel = mlOps(X_train, X_test, y_train, y_test, model, name)
    print(name+" done")

    name = "AdaBoostClassifier"
    model = AdaBoostClassifier()
    AdaBdct, AdaBmodel = mlOps(X_train, X_test, y_train, y_test, model, name)
    print(name+" done")

    mdct = {"NBmodel":NBmodel,"LGBMmodel":LGBMmodel,"XGBmodel":XGBmodel,"AdaBmodel":AdaBmodel}
    metrices_dct = {"NBdct":NBdct,"LGBMdct":LGBMdct,"XGBdct":XGBdct,"AdaBdct":AdaBdct}

    return mdct, metrices_dct