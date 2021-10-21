import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from transformers import DistilBertTokenizer, RobertaTokenizer
from xgboost import XGBClassifier

from models import JobsLevelPredictBertModel, JobsLevelPredictRoBertModel
from parameter_tuning import tune
from text_preprocess import TFIDFTokenize, tokenize_and_transform
from train import train_transformer, train_ml_models
from utils import load_data, create_test_set, create_folds
from test import test_model

sklearn_models = {
    "naive_bayes": MultinomialNB(),
    "random_forest": RandomForestClassifier(),
    "xgb_model": XGBClassifier(objective='multi:softprob',
                               use_label_encoder=False,
                               n_estimators=50)
}

transformer_models = {
    "bert_model": JobsLevelPredictBertModel(),
    "roberta": JobsLevelPredictRoBertModel()
}


def parse_args():
    parser = argparse.ArgumentParser(prog='job-level-fill',
                                     description='Predicting Job seniority level',
                                     argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--model',
        help="Model name",
        choices=["naive_bayes", "random_forest",
                 "bert",
                 "roberta",
                 "xgb_model"]
    )
    parser.add_argument(
        '--pre-process',
        help='Pre-process type',
        choices=["tfidf", "embedding"])

    parser.add_argument(
        '--tune',
        action="store_true",
        help='Hyper parameter tuning')

    parser.add_argument(
        '--test',
        action="store_true",
        help='Evaluation on the test set.')

    args = parser.parse_args()
    return args


def main(args):
    """
    Please run the train and test methods with the correct arguments to train and test the models.
    We have both Machine learning models and Transformer models. Machine learning models support two pre-processing types,
    one is `tfidf` and the other is `word embeddings`.
    Use the following syntax to train and model for both pre-processings.
    There is a limitation in using embedding processing for Naive Bayes model because it assumes data should not have
    negative values but in our glove embeddings have negative values so there is not support of embedding pre-processing for Naive Bayes model.

    train_ml_models(model,
                train_data=train_data,
                model_name=model_name,
                pre_processing_fun=(tfidf_process|tokenize_and_transform),
                pre_processing="tfidf|embedding")

    Example
    train_ml_models(naive_bayes, train_data, 'NaiveBayes', pre_processing_fun=tfidf_process, pre_processing='embedding')

    """
    data = load_data()
    train_data, test_data = create_test_set(data)
    train_data = create_folds(train_data)
    tfidf_process = TFIDFTokenize()
    tfidf_process.fit(train_data[['description', 'title']], train_data["level"])

    # Train sklearn models.
    if args["model"] in sklearn_models.keys() and \
            'pre_process' in args and \
            args["pre_process"] == "tfidf":

        train_ml_models(sklearn_models[args["model"]], train_data, args["model"],
                        pre_processing_fun=tfidf_process, pre_processing='tfidf')

    elif args["model"] in sklearn_models.keys() and \
            'pre_process' in args and \
            args["pre_process"] == "embedding":

        train_ml_models(sklearn_models[args["model"]], train_data, args["model"],
                        pre_processing_fun=tokenize_and_transform, pre_processing='embedding')

    elif args["model"] in transformer_models.keys():
        # Train Transformer models
        if args["model"] == "roberta":
            roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            train_transformer(transformer_models[args["model"]],
                              train_data=train_data,
                              tokenizer=roberta_tokenizer,
                              learning_rate=1e-3, epochs=50,
                              model_name=args["model"])
        elif args["model"] == "bert":
            distilled_bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            train_transformer(transformer_models[args["model"]],
                              train_data=train_data,
                              tokenizer=distilled_bert_tokenizer,
                              learning_rate=1e-3, epochs=50,
                              model_name=args["model"])

    elif "tune" in args:
        # Hyper parameter tuning
        tune(XGBClassifier, train_data, tokenize_and_transform)

    elif "test" in args:
        test_model(args["model"],
                   test_data=test_data,
                   pre_processing_fun=tokenize_and_transform)


if __name__ == "__main__":
    args = vars(parse_args())
    print(args)
    main(args)
