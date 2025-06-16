import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow

def svm_train(data, target_column, test_size=0.2, random_state=42):
    # Setup experiment
    mlflow.set_experiment("Loan_Approval_SVM")
    
    # Split data
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Start run
    with mlflow.start_run(run_name="SVM_Loan_Approval"):
        # Aktifkan autolog
        mlflow.autolog()
        
        # Train model
        model = SVC(probability=True, random_state=random_state)
        model.fit(X_train, y_train)
        
        # Evaluasi 
        accuracy = model.score(X_test, y_test)
        
    return model, accuracy

if __name__ == "__main__":

    data = pd.read_csv("loan_preprocessing.csv")
    
    model, acc = svm_train(
        data, 
        target_column="loan_status",
        test_size=0.2,
        random_state=42
    )
    
    print(f"Final accuracy: {acc:.2%}")