import reprlib


def customize_type(**default_kwargs):
    def deco(type_):
        type_._custom_default_kwargs = default_kwargs
        return type_
    return deco


class Config(object):

    def __init__(self, config, types=None):
        self._config = config
        self._types = types or {}
        self._cache = {}

    def __getitem__(self, key):
        return self._eval(*_parse_key(key, None), ())

    def _eval(self, config_key, attr_key, trace):
        circular = (config_key, attr_key) in trace
        trace = (*trace, (config_key, attr_key))
        if circular:
            raise RuntimeError('Circular dependency: {}'.format(
                ' -> '.join(_dump_key(config_key, attr_key)
                            for config_key, attr_key in trace)))

        obj = self._eval_config(config_key, trace)
        try:
            for k in attr_key:
                if isinstance(k, str) and hasattr(obj, k):
                    obj = getattr(obj, k)
                else:
                    obj = obj[k]
        except (IndexError, KeyError):
            raise KeyError('{} does not exist: {}'.format(
                _dump_key(config_key, attr_key),
                ' -> '.join(_dump_key(config_key, attr_key)
                            for config_key, attr_key in trace)))

        return obj

    def _eval_config(self, config_key, trace):
        if config_key in self._cache:
            return self._cache[config_key]

        config = self._config
        try:
            for k in config_key:
                config = config[k]
        except (IndexError, KeyError):
            raise KeyError('{} does not exist: {}'.format(
                _dump_key(config_key, ()),
                ' -> '.join(_dump_key(config_key, attr_key)
                            for config_key, attr_key in trace)))

        if isinstance(config, dict):
            if 'type' in config:
                type_ = self._types[config['type']]
            else:
                type_ = dict

            kwargs = {}
            for k in config.keys():
                if not k == 'type':
                    kwargs[k] = self._eval_config((*config_key, k), trace)
            for k, v in getattr(type_, '_custom_default_kwargs', {}).items():
                if k not in kwargs:
                    kwargs[k] = self._eval(*_parse_key(v, config_key), trace)

            try:
                self._cache[config_key] = type_(**kwargs)
            except Exception as e:
                if len(e.args) > 0:
                    e.args = ('{} (type_ = {}, kwargs = {})'.format(
                        e.args[0], type_, reprlib.repr(kwargs)), *e.args[1:])
                else:
                    e.args = ('(type_ = {}, kwargs = {})'.format(
                        type_, reprlib.repr(kwargs)),)
                raise e

        elif isinstance(config, list):
            self._cache[config_key] = [
                self._eval_config((*config_key, i), trace)
                for i in range(len(config))]
        else:
            if isinstance(config, str) and config.startswith('@'):
                self._cache[config_key] = self._eval(
                    *_parse_key(config[1:], config_key[:-1]), trace)
            else:
                self._cache[config_key] = config

        return self._cache[config_key]


def _parse_key(key, path):
    if key.startswith('/'):
        key = key[1:]
        rel = False
    else:
        rel = True
    assert not rel or path is not None

    config_key = key.split('/')
    config_key[-1], *attr_key = config_key[-1].split('.')

    config_key = [_parse_k(k) for k in config_key]
    attr_key = tuple(_parse_k(k) for k in attr_key)

    if rel:
        config_key = list(path) + config_key

    i = 0
    while i < len(config_key):
        if config_key[i] in {'', '.'}:
            config_key.pop(i)
        elif config_key[i] == '..':
            assert i > 0
            config_key.pop(i)
            config_key.pop(i - 1)
            i -= 1
        else:
            i += 1

    return tuple(config_key), attr_key


def _parse_k(k):
    try:
        return int(k)
    except ValueError:
        return k


def _dump_key(config_key, attr_key):
    config_key = '/' + '/'.join(str(k) for k in config_key)
    attr_key = '.'.join(str(k) for k in attr_key)

    if attr_key:
        return config_key + '.' + attr_key
    else:
        return config_key
