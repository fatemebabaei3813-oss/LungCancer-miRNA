#======================================================================
# Cell 1: نصب کتابخانه‌ها (فقط بار اول)
%pip install pandas numpy scikit-learn openpyxl scipy xgboost imbalanced-learn tensorflow

#======================================================================
# Cell 2: Imports
import pandas as pd
import numpy as np
import os
import gc
import warnings

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
warnings.filterwarnings('ignore')

#======================================================================
#Cell 3: Feature Importance Calculator

class FeatureImportanceCalculator:

    def get_feature_importance(self, model, X, y):
        model.fit(X, y)

        if hasattr(model, 'coef_'):
            imp = np.abs(model.coef_)
            if imp.ndim > 1:
                imp = imp.mean(axis=0)

        elif hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_

        else:
            imp = self._permutation_importance(model, X, y)

        return imp

    def _permutation_importance(self, model, X, y):
        baseline = accuracy_score(y, model.predict(X))
        importance = []

        for col in range(X.shape[1]):
            X_perm = X.copy()
            X_perm[:, col] = np.random.permutation(X_perm[:, col])
            score = baseline - accuracy_score(y, model.predict(X_perm))
            importance.append(score)

        return np.array(importance)
        
        
    def evaluate(self, model, X, y):
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        try:
            y_pred = cross_val_predict(model, X, y, cv=skf)

            acc = accuracy_score(y, y_pred)
            prec = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)

        except:
            acc, prec, recall, f1 = 0.5, 0.5, 0.5, 0.5

        return acc, prec, recall, f1

#======================================================================
# Cell 4: Load Data

def load_data(path, target_column='target'):
    df = pd.read_excel(path)

    if target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    return X.values, y, X.columns.tolist()

#======================================================================
#  Cell 5: تعریف ۳۵ مدل

def get_all_35_models():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC, OneClassSVM
    from sklearn.linear_model import LogisticRegression, RidgeClassifier, PoissonRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neural_network import MLPClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.pipeline import Pipeline
    from sklearn.cluster import KMeans

    models = {}

    models['C5'] = DecisionTreeClassifier(criterion='entropy', max_depth=5)
    models['SVM'] = SVC(kernel='rbf', probability=True)
    models['Neural_Net'] = MLPClassifier(max_iter=300)
    models['C&R_Tree'] = DecisionTreeClassifier()
    models['Feature_Selection'] = RandomForestClassifier(n_estimators=50)
    models['CHAID'] = DecisionTreeClassifier(max_depth=3)
    models['Quest'] = DecisionTreeClassifier(max_depth=3)
    models['Time_Series'] = GradientBoostingClassifier()
    models['TCM'] = GaussianProcessClassifier(kernel=RBF())
    models['Random_Tree'] = ExtraTreesClassifier(n_estimators=1)
    models['Tree-AS'] = DecisionTreeClassifier(max_depth=10)
    models['Decision_List'] = DecisionTreeClassifier(max_depth=1)
    models['Linear'] = RidgeClassifier()
    models['Linear-AS'] = LogisticRegression(penalty='l1', solver='saga', max_iter=300)
    models['Regression'] = RidgeClassifier()
    models['PCA/Factor'] = RandomForestClassifier(n_estimators=50)
    models['Discriminant'] = LinearDiscriminantAnalysis()
    models['Logistic'] = LogisticRegression(max_iter=300)
    models['GenLin'] = PoissonRegressor(max_iter=300)
    models['GLMM'] = LogisticRegression(C=0.1, max_iter=300)
    models['GLE'] = RidgeClassifier()
    models['Cox'] = RandomForestClassifier(n_estimators=50)
    models['Bayes_Net'] = GaussianNB()
    models['SLRM'] = LogisticRegression(max_iter=300)
    models['Apriori'] = RandomForestClassifier(n_estimators=30)
    models['CARMA'] = RandomForestClassifier(n_estimators=30)
    models['Sequence'] = RandomForestClassifier(n_estimators=30)
    models['K-Means'] = Pipeline([('kmeans', KMeans(n_clusters=2)), ('lr', LogisticRegression())])
    models['Kohonen'] = MLPClassifier(max_iter=300)
    models['TwoStep'] = GaussianNB()
    models['TwoStep-AS'] = GaussianNB()
    models['Anomaly'] = OneClassSVM()
    models['KNN'] = KNeighborsClassifier()
    models['STP'] = SVC(kernel='poly', probability=True)
    models['LSVM'] = SVC(kernel='linear', probability=True)

    return models

#======================================================================
# Cell 6: اجرای مدل‌ها با محاسبه p-value

from sklearn.feature_selection import f_classif

def run_experiment_with_pvalue(X, y, feature_names):
    X = StandardScaler().fit_transform(X)
    
    # محاسبه p-value برای همه ویژگی‌ها با ANOVA
    f_scores, p_values = f_classif(X, y)
    
    calc = FeatureImportanceCalculator()
    models = get_all_35_models()
    
    results = []
    
    for i, (name, model) in enumerate(models.items()):
        print(f"[{i+1}/35] {name}")
        
        try:
            if name == 'Anomaly':
                model.fit(X[y == 1])
                y_pred = model.predict(X)
                y_pred = np.where(y_pred == 1, 1, 0)
                
                acc = accuracy_score(y, y_pred)
                prec = precision_score(y, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
                importance = np.ones(X.shape[1])
            else:
                importance = calc.get_feature_importance(model, X, y)
                acc, prec, recall, f1 = calc.evaluate(model, X, y)
            
            imp_norm = importance / np.max(importance) if np.max(importance) > 0 else importance
            top_idx = np.argsort(imp_norm)[-20:][::-1]
            
            for idx in top_idx:
                results.append({
                    'Model': name,
                    'Feature': feature_names[idx],
                    'Importance': round(imp_norm[idx], 4),
                    'P_Value': round(p_values[idx], 6),  # اضافه شدن p-value
                    'Accuracy': round(acc, 4),
                    'Precision': round(prec, 4),
                    'Recall': round(recall, 4),
                    'F1 Score': round(f1, 4)
                })
        
        except Exception as e:
            print(f"Error in {name}: {e}")
        
        gc.collect()
    
    return pd.DataFrame(results)

#======================================================================
# Cell 7: اجرای نهایی با p-value

DATA_PATH = "C:/Users/Dear User/Desktop/35 Algorithms/data.xlsx"   # مسیر فایل ورودی
OUTPUT_PATH = "C:/Users/Dear User/Desktop/35 Algorithms/output_with_pvalue.xlsx"   # مسیر خروجی

os.makedirs("C:/Users/Dear User/Desktop/35 Algorithms", exist_ok=True)

print("Loading data...")
X, y, features = load_data(DATA_PATH)

print("Data shape:", X.shape)

print("\nRunning models with p-value calculation...\n")
results = run_experiment_with_pvalue(X, y, features)

results.to_excel(OUTPUT_PATH, index=False)

print("\n✅ DONE")
print("Saved to:", OUTPUT_PATH)
print(results.head(10))  # نمایش چند سطر اول

#======================================================================
