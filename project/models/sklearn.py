from typing import Union
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle


class SklearnClassifier:
    def __init__(
        self, 
        classifier: Union[KNeighborsClassifier, RandomForestClassifier], 
        train_data: tuple[np.ndarray, np.ndarray], 
        val_data: tuple[np.ndarray, np.ndarray], 
        test_data: tuple[np.ndarray, np.ndarray]
    ):
        self.classifier = classifier
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def fit(self, model_name: Union[str, None] = None) -> None:
        self.classifier.fit(*self.train_data)
        if model_name:
            self.pickle_model(model_name=model_name)
            print(f'Saved {model_name} to pickled_objects/{model_name}.pkl')

    def val_acc_score(self) -> float:
        y_pred = self.classifier.predict(self.val_data[0])
        return accuracy_score(y_true=self.val_data[-1], y_pred=y_pred)
    
    def test_acc_score(self) -> float:
        y_pred = self.classifier.predict(self.test_data[0])
        return accuracy_score(y_true=self.test_data[-1], y_pred=y_pred)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classifier.predict(X)
    
    def pickle_model(self, model_name: str) -> None:
        pickle_dir = 'pickled_objects'
        with open(f"{pickle_dir}/{model_name}.pkl", 'wb') as f:
            pickle.dump(self.classifier, f)


def load_model(model_name: str) -> Union[KNeighborsClassifier, RandomForestClassifier]:
    """Utility function to get pickled model."""
    with open(f"pickled_objects/{model_name}.pkl", 'rb') as f:
        return pickle.load(f)