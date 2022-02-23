def get_features_and_labels_names(models):
    features = models[0].get_input_features_names()
    labels = models[0].get_outputs_names()
    for model in models[1:]:
        features.extend(model.get_input_features_names())
        labels.extend(model.get_outputs_names())
    features = list(set(features))
    labels = list(set(labels))
    return features, labels



