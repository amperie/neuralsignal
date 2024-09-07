import logging
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import Apply
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import time
from neuralsignal.core.modules.neuralsignal_config import sdk_config
from neuralsignal.backend.ns_backend import NSBackend
from neuralsignal.core.modules.s1_model import S1Model
from neuralsignal.core.modules.utils import dict_to_str

logging.basicConfig(level=sdk_config.logging_level())


class S1Trainer:
    """
    Trains an S1 model from a dataset
    """

    default_config = {
        "application_name": None,
        "sub_application_name": None,
        "model_name": None,
        "description": None,
        "save_to_backend": True,
        "dataset_path": None,
        "columns_to_drop": [],
        "row_limit": 0,
        "seed": 42,
        "test_size": 0.33,
        "hyperopt_space": {
            'max_depth': hp.uniform("max_depth", 3, 18),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
            'max_bin': hp.uniform('max_bin', 32, 512),
            'n_estimators': hp.uniform('n_estimators', 500, 5000),
        },
        "device": "cuda",
        "test_set_size": 0.33,
        "optimization_metric": "auc",
        "xgboost_optimization_metric": "logloss",
        "prediction_threshold": 0.5,
        "max_evals": 2,
        "early_stopping_rounds": 50,
        "metrics": {},
        "params": {},
        "tags": {},
        "metadata": {},
        "create_reduced_feature_model": False,
        "reduced_feature_count": 20,
    }

    def configure_hyperopt_space(self, config):
        # Hyperopt config from yaml should be:
        # {"param_name": [min_val, max_val]}
        hos = config['hyperopt_space']
        hoc = {}
        for k, v in hos.items():
            hoc[k] = hp.uniform(k, v[0], v[1])
        self.config['hyperopt_space'] = hoc

    def __init__(self, config) -> None:

        # Application_name and sub_name are required
        if "application_name" not in config:
            raise ValueError("Missing application_name in config")
        if "sub_application_name" not in config:
            raise ValueError("Missing sub_application_name in config")
        if "model_name" not in config:
            raise ValueError("Missing model_name in config")
        self.config = {**self.default_config, **config}
        # self.original_config = copy.deepcopy(self.config)
        # Check if dataset path is set
        if self.config['dataset_path'] is None:
            raise ValueError("Dataset path is not set")

        if "hyperopt_space" in config and not\
                isinstance(config['hyperopt_space']['max_depth'], Apply):
            self.configure_hyperopt_space(config)
        logging.debug(
            f"Hyperopt space: {dict_to_str(self.config['hyperopt_space'])}"
            )
        self.data_loaded = False
        self.metrics = self.config["metrics"]
        self.params = self.config["params"]
        self.tags = self.config["tags"]
        self.metadata = self.config["metadata"]
        self.params['dataset_path'] = self.config['dataset_path']
        logging.info(f"Initialized S1Trainer with config: {self.config}")
        # The whole dataset
        self.X = None
        self.Y = None
        # Train and test partitions
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data_loaded = False
        self.best_model = None

        self.be = NSBackend(config)

    def load_data_from_dataframe(self, data: pd.DataFrame) -> None:

        if self.data_loaded:
            return

        # Drop any extra columns that have formed
        data = data.dropna(axis=1, how='all')

        if self.config['row_limit'] > 0:
            logging.info(
                f"Reading dataset {self.config['dataset_path']} "
                f"with row limit: {self.config['row_limit']}")
            data = data.head(self.config['row_limit'])

        self.X = data.drop(self.config['columns_to_drop'], axis=1)
        self.X = self.X.drop(['target'], axis=1)
        self.Y = data['target']

        seed = self.config['seed']
        self.X_train, self.X_test, self.y_train, self.y_test =\
            train_test_split(
                self.X, self.Y, test_size=self.config['test_set_size'],
                random_state=seed
                )

        self.data_loaded = True
        logging.info(
            f"Dataset {self.config['dataset_path']} shape: {data.shape}")

    def load_splits_from_dataframe(
            self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        if self.data_loaded:
            return

        self.X_train = train_data.drop(self.config['columns_to_drop'], axis=1)
        self.X_train = self.X_train.drop(['target'], axis=1)
        self.y_train = train_data['target']
        self.X_test = test_data.drop(self.config['columns_to_drop'], axis=1)
        self.X_test = self.X_test.drop(['target'], axis=1)
        self.y_test = test_data['target']

        self.X = pd.concat([self.X_train, self.X_test])
        self.Y = pd.concat([self.y_train, self.y_test])

        self.data_loaded = True
        logging.info(
            f"Loaded splits directly with shape: {self.X.shape}")

    def load_data(self) -> None:
        if self.data_loaded:
            return

        if self.config['row_limit'] > 0:
            logging.info(
                f"Reading dataset {self.config['dataset_path']} "
                f"with row limit: {self.config['row_limit']}")
            data = pd.read_csv(
                self.config['dataset_path'], nrows=self.config['row_limit'])
        else:
            logging.info(f"Reading dataset {self.config['dataset_path']}")
            data = pd.read_csv(self.config['dataset_path'])

        self.load_data_from_dataframe(data)

    def _create_reduced_feature_model(self, feature_list: list):

        cfg = copy.deepcopy(self.config)
        # cfg.pop("dataframe")
        cfg['params']['feature_list'] = feature_list
        cfg['params']['reduced_feature_model'] = True
        cfg['create_reduced_feature_model'] = False
        mt = S1Trainer(cfg)

        train_data = pd.concat(
            [self.y_train, self.X_train[feature_list]], axis=1)

        test_data = pd.concat(
            [self.y_test, self.X_test[feature_list]], axis=1)

        mt.load_splits_from_dataframe(train_data, test_data)
        m = mt.train_model()
        return m

    def _hyperparameter_tuning(self, space):

        model = xgb.XGBClassifier(
            eval_metric=self.config['xgboost_optimization_metric'],
            early_stopping_rounds=self.config['early_stopping_rounds'],
            n_estimators=int(space['n_estimators']),
            max_depth=int(space['max_depth']),
            max_bin=int(space['max_bin']), device=self.config['device'],
            tree_method='hist')

        evaluation = [(self.X_train, self.y_train), (self.X_test, self.y_test)]

        model.fit(
            self.X_train, self.y_train,
            eval_set=evaluation)

        pred = model.predict(self.X_test)
        pred_proba = model.predict_proba(self.X_test)[:, 1]
        match self.config['optimization_metric']:
            case "accuracy":
                accuracy = accuracy_score(
                    self.y_test, pred > self.config['prediction_threshold'])
                return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}
            case "auc":
                auc = roc_auc_score(self.y_test, pred_proba)
                return {'loss': -auc, 'status': STATUS_OK, 'model': model}
            case "TN":
                tn = confusion_matrix(self.y_test, pred)[0, 0]
                return {'loss': -tn, 'status': STATUS_OK, 'model': model}
            case "TP":
                tp = confusion_matrix(self.y_test, pred)[1, 1]
                return {'loss': -tp, 'status': STATUS_OK, 'model': model}
            case "FP":
                fp = confusion_matrix(self.y_test, pred)[0, 1]
                return {'loss': fp, 'status': STATUS_OK, 'model': model}
            case "TP+TN":
                cm = confusion_matrix(self.y_test, pred)
                tp = cm[1, 1]
                tn = cm[0, 0]
                return {
                    'loss': -(tp + tn), 'status': STATUS_OK, 'model': model}
            case _:
                ll = log_loss(self.y_test, pred_proba)
                return {'loss': ll, 'status': STATUS_OK, 'model': model}

    def train_model(self) -> S1Model:
        self.load_data()
        logging.info("Starting hyperparameter tuning")
        trials = Trials()

        # Run hyperparameter tuning
        tic = time.perf_counter()
        best = fmin(
            fn=self._hyperparameter_tuning,
            space=self.config['hyperopt_space'],
            algo=tpe.suggest,
            max_evals=self.config['max_evals'],
            trials=trials)
        toc = time.perf_counter()

        logging.info(
            f"Hyperparameter tuning took {toc - tic:0.4f} seconds")

        # Store parameters and best model
        self.best_params = best
        self.best_params['n_estimators'] =\
            int(self.best_params['n_estimators'])
        self.best_params['max_depth'] = int(self.best_params['max_depth'])
        self.best_params['max_bin'] = int(self.best_params['max_bin'])
        self.params.update(best)
        self.best_model = trials.best_trial['result']['model']
        model = self.best_model
        logging.info(
            f"Best model params: {self.best_params}")
        logging.info(
            f"Best model: {self.best_model}")

        pred = model.predict_proba(self.X_train)[:, 1] >= \
            self.config['prediction_threshold']
        pred_proba = model.predict_proba(self.X_train)[:, 1]

        # Calculate metrics
        train_accuracy = accuracy_score(self.y_train, pred)
        train_auc = metrics.roc_auc_score(self.y_train, pred_proba)
        train_ll = log_loss(self.y_train, pred_proba)
        logging.info(
            f"Training accuracy: {train_accuracy}")
        logging.info(
            f"Training auc: {train_auc}")
        logging.info(
            f"Training log loss: {train_ll}")
        self.metrics['train_accuracy'] = train_accuracy
        self.metrics['train_auc'] = train_auc
        self.metrics['train_ll'] = train_ll
        self.metrics['rows_training'] = len(self.X_train)

        # Calculate metrics on test set
        pred = model.predict_proba(self.X_test)[:, 1] >= \
            self.config['prediction_threshold']
        pred_proba = model.predict_proba(self.X_test)[:, 1]
        test_accuracy = accuracy_score(self.y_test, pred)
        test_auc = metrics.roc_auc_score(self.y_test, pred_proba)
        test_ll = log_loss(self.y_test, pred_proba)
        logging.info(
            f"Test accuracy: {test_accuracy}")
        logging.info(
            f"Test auc: {test_auc}")
        logging.info(
            f"Test log loss: {test_ll}")
        self.metrics['test_accuracy'] = test_accuracy
        self.metrics['test_auc'] = test_auc
        self.metrics['test_ll'] = test_ll
        self.metrics['rows_test'] = len(self.X_test)
        self.metrics['rows_dataset'] = len(self.X)

        self.metrics['dataset_column_count'] = len(self.X.columns)

        # Make sure to save other params
        self.params['optimization_metric'] = self.config['optimization_metric']
        self.params['max_evals'] = self.config['max_evals']
        self.params['prediction_threshold'] = \
            self.config['prediction_threshold']
        self.params['early_stopping_rounds'] = \
            self.config['early_stopping_rounds']
        self.params['features_dimension'] = \
            len(self.X.columns)

        model_cfg = {
            "application_name": self.config['application_name'],
            "sub_application_name": self.config['sub_application_name'],
            "model_name": self.config['model_name'],
            "model_id": None,
            "description": self.config['description'],
            "dataset_path": self.config['dataset_path'],
            "optimization_metric": self.config['optimization_metric'],
            'metrics': self.metrics,
            'params': self.params,
            'metadata': self.metadata,
            'tags': self.tags,
            "model": self.best_model
        }

        # Feature Importance
        create_reduced_model = self.config['create_reduced_feature_model']
        fi_to_save = self.config['reduced_feature_count']

        fi =\
            self.best_model.get_booster().get_score(importance_type='gain')
        fi_s = dict(
            sorted(fi.items(), key=lambda x: x[1], reverse=True)[0:20])
        model_cfg['artifacts'] = {'feature_importance': fi_s}

        if create_reduced_model:
            fi_s = dict(
                sorted(
                    fi.items(), key=lambda x: x[1],
                    reverse=True)[0:fi_to_save])
            feature_list = list(fi_s.keys())
            self._create_reduced_feature_model(feature_list)

        if "run_name" in self.config:
            model_cfg['run_name'] = self.config['run_name']

        # Make plots
        # TODO: this
        cm = confusion_matrix(
            self.y_test,
            pred_proba >= self.config['prediction_threshold'],
            labels=model.classes_)
        conf_matrix = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=model.classes_)
        conf_matrix.plot()
        cm_fig = conf_matrix.figure_
        model_cfg['figures'] = {'confusion_matrix': cm_fig}
        model_cfg['params']['confusion_matrix'] = str(cm)

        model_to_save = S1Model(model_cfg)
        if self.config['save_to_backend']:
            model_to_save = self.be.save_s1_model(model_to_save)

        logging.info(
            f"Training completed for model {self.config['model_name']} "
            f"in application {self.config['application_name']} "
            f"sub application {self.config['sub_application_name']}"
            )
        return model_to_save
