import joblib

label_mappings = {
    0: 'Entry Level',
    1: 'Internship',
    2: 'Mid Level',
    3: 'Senior Level'
}

def test_model(model_name, test_data, pre_processing_fun, processing_type="embedding"):
    """
    Test the final model performance on the test set
    :param model_name:
    :param test_data:
    :param pre_processing_fun:
    :param processing_type:
    :return:
    """
    test_data = test_data[test_data.description != "no"]
    X_test, _ = pre_processing_fun(test_data)
    model = joblib.load(f"../models/{model_name}_{processing_type}.pkl")
    predictions = model.predict(X_test)
    test_data["level"] = predictions
    test_data["level"] = test_data.level.apply(lambda x: label_mappings[x])
    test_data.to_json("../output/predictions.json", orient="records")