from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import os


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

def pred_success(X,channel_id):
    direc = "models/"+channel_id+"/"

    joblib_file = direc+"GaussianNB"+".pkl"
    GaussianNB = joblib.load(joblib_file)
    joblib_file = direc+"LGBMClassifier"+".pkl"
    LGBMClassifier = joblib.load(joblib_file)
    joblib_file = direc+"XGBClassifier"+".pkl"
    XGBClassifier = joblib.load(joblib_file)
    joblib_file = direc+"AdaBoostClassifier"+".pkl"
    AdaBoostClassifier = joblib.load(joblib_file)

    preds = {"GaussianNB":GaussianNB.predict(X),
            "LGBMClassifier":LGBMClassifier.predict(X),
            "XGBClassifier":XGBClassifier.predict(X),
            "AdaBoostClassifier":AdaBoostClassifier.predict(X)}
    
    return preds

def clf_main(X_train, X_test, y_train, y_test,channel_id):
    direc = "models/"+channel_id+"/"

    if not os.path.exists("models/"):
        os.mkdir("models/")
    if not os.path.exists("models/"+channel_id+"/"):
        os.mkdir(direc)

    name = "GaussianNB"
    model = GaussianNB()
    NBdct, NBmodel = mlOps(X_train, X_test, y_train, y_test, model, name)
    print(name+" done")
    joblib_file = direc+name+".pkl"
    joblib.dump(NBmodel, joblib_file)

    name = "LGBMClassifier"
    model = LGBMClassifier()
    LGBMdct, LGBMmodel = mlOps(X_train, X_test, y_train, y_test, model, name)
    print(name+" done")
    joblib_file = direc+name+".pkl"
    joblib.dump(NBmodel, joblib_file)

    name = "XGBClassifier"
    model = XGBClassifier()
    XGBdct, XGBmodel = mlOps(X_train, X_test, y_train, y_test, model, name)
    print(name+" done")
    joblib_file = direc+name+".pkl"
    joblib.dump(NBmodel, joblib_file)

    name = "AdaBoostClassifier"
    model = AdaBoostClassifier()
    AdaBdct, AdaBmodel = mlOps(X_train, X_test, y_train, y_test, model, name)
    print(name+" done")
    joblib_file = direc+name+".pkl"
    joblib.dump(NBmodel, joblib_file)

    mdct = {"NBmodel":NBmodel,"LGBMmodel":LGBMmodel,"XGBmodel":XGBmodel,"AdaBmodel":AdaBmodel}
    metrices_dct = {"NBdct":NBdct,"LGBMdct":LGBMdct,"XGBdct":XGBdct,"AdaBdct":AdaBdct}

    return mdct, metrices_dct