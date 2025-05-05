import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("CreditInfo.csv")
df[df.select_dtypes("int64").columns] = df[df.select_dtypes("int64").columns].astype("float64")

# Select variables
y = df['Credit_Mix'].to_numpy()
X = df.iloc[:, df.columns != 'Credit_Mix'].to_numpy()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = RandomForestClassifier()
#
# parameters = {"max_depth": range(2,13)}
# grid_search = GridSearchCV(clf, param_grid=parameters, cv=5)
# grid_search.fit(X_train, y_train)
# score_df = pd.DataFrame(grid_search.cv_results_)
#print(score_df[['param_max_depth','mean_test_score','rank_test_score']])

# max_depth = grid_search.best_params_["max_depth"]
# print(max_depth)


clf = RandomForestClassifier(max_depth=10, oob_score=True, verbose=3)
clf.fit(X_train, y_train)

importances = pd.DataFrame(clf.feature_importances_, index=df.columns[1:])
importances.plot.bar()
plt.show()

print(f"Score (Train): {clf.score(X_train, y_train):.3f}")
print(f"Score (Test): {clf.score(X_test, y_test):.3f}")
print(f"OOB Score: {clf.oob_score_:.3f}")

cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()
