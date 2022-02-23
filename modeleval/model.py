import json

import awswrangler
import mlflow
import yaml
from mlflow.tracking import MlflowClient


class Model:
    def __init__(self, input_model_name, tag=None):
        model_name, version = self.get_model_name_version(input_model_name)

        self.input_name = input_model_name
        self.name = model_name
        self.version = version
        self.tag = tag

        client = MlflowClient()
        model_version = client.get_model_version(
            model_name, version
        )
        self.model = mlflow.lightgbm.load_model(model_version.source)
        ml_model_path = "MLmodel.yaml"
        awswrangler.s3.download(
            f"{model_version.source}/MLmodel", ml_model_path)

        with open(ml_model_path, "r") as f:
            signature = yaml.load(
                f, Loader=yaml.FullLoader)["signature"]

            self.inputs = json.loads(signature["inputs"])
            self.outputs = json.loads(signature["outputs"])

    def get_tag(self):
        if self.tag is not None:
            return self.tag
        return self.input_name

    def get_model_name_version(self, model_string):
        model_parts = model_string.split('.')
        if len(model_parts) != 3:
            return None, None
        return model_parts[0] + '-' + model_parts[1], model_parts[2]

    def get_input_features_names(self):
        return [input['name'] for input in self.inputs]

    def get_outputs_names(self):
        return [output['name'] for output in self.outputs]

    def prediction(self, df):
        columns = df.columns
        inputs_names = self.get_input_features_names()
        signature_flag = True
        for i in inputs_names:
            if i not in columns:
                signature_flag = False
        if signature_flag:
            features = df[[input for input in inputs_names]]
            return self.model.predict(features)
        return None
