from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def build_pipeline(preprocessor):
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", LogisticRegression(
            max_iter = 2000,
            class_weight = "balanced"
        ))
    ])
    return pipeline