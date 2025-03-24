#this script will retrain an AI model and save the updated vresion periodically
#To Schedule the script to run automatically in Windows
#Win + r -> type taskschd.msc -> enter
#Create a new task -> create a basic task -> name: Any name -> daily -> time
#set Action -> Start a program -> browse for python.exe 
# -> add arguments: set the path to the model.py script -> Finish
#verify the task -> open task scheduler and look under active tasks

import joblib
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save updated model with timestamp
#extract date and time from the system
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#set the file name with the timestamp
model_filename = f'updated_model_{timestamp}.pkl'
#save the model
joblib.dump(model, model_filename)

print(f"Model retrained and saved as {model_filename}")