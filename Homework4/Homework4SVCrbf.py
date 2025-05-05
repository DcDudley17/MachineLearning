import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.svm import SVC



# Load data
df = pd.read_csv("weather_classification_data_weatherType.csv")
df.dropna(inplace=True)

# Take a random sample of 5000 rows
df = df.sample(n=5000)

# Ensure correct data types
df[df.select_dtypes("int64").columns] = df[df.select_dtypes("int64").columns].astype("float64")

# Features and 'season' is the target column
X = df.drop(columns=['Weather Type']).copy().to_numpy()
y = df['Weather Type'].astype('category').to_numpy()

# normalize the data
X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

# Train/test split with 30% left as a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = SVC(kernel="rbf")

# Set up the parameter grid for both C and gamma
param_grid = {
    "C": np.linspace(1, 500, 10),
    "gamma": np.linspace(0.01,0.1,5)
}
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_C', 'param_gamma', 'mean_test_score', 'rank_test_score']])


#The larger C is, the closer to the hard margin we get
C = grid_search.best_params_["C"]
gamma = grid_search.best_params_["gamma"]
print(C)
print(gamma)

clf = SVC(C = 1.0,kernel="rbf")


clf.fit(X_train, y_train)
print(f"Score: {clf.score(X_test, y_test):.3f}")


#Why not use score?
#If we have an imbalanced dataset with one category having significantly more instances than the other
#This would result in us being able to predict the majority case consistency but fail on minority cases
#In my dataset, there is an equal balance of weather types, so using the score is an accurate measure

cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()

