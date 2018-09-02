from collections import defaultdict

from .rcnn import RCNN
from .simple_lstm import SimpleLSTM


available_models = {
    'simple_lstm': SimpleLSTM,
    'rcnn': RCNN
}


def build_model(options, fields):
    dict_fields = defaultdict(lambda: None)
    dict_fields.update(dict(fields))
    model_class = available_models[options.model]
    model = model_class(dict_fields['words'],
                        dict_fields['tags'],
                        prefixes_field=dict_fields['prefixes'],
                        suffixes_field=dict_fields['suffixes'],
                        caps_field=dict_fields['caps'])
    model.build(options)
    return model
