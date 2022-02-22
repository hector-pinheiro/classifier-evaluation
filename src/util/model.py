def get_features_and_labels(models):
    features = models[0].get_input_features_names()
    labels = models[0].get_outputs_names()
    for model in models[1:]:
        features.extend(model.get_input_features_names())
        labels.extend(model.get_outputs_names())
    features = list(set(features))
    labels = list(set(labels))
    return features, labels


def get_model_name_version(model):
    model_parts = model.split('.')
    if len(model_parts) != 3:
        return None, None
    return model_parts[0] + '-' + model_parts[1], model_parts[2]
