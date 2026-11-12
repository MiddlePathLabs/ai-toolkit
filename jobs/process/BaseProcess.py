import copy
import json
import re
from collections import OrderedDict

from toolkit.timer import Timer

# api_key, apikey, hf_token, client_secret, password ... but not tokenizer / max_new_tokens
_SECRET_KEY_RE = re.compile(r"(^|_)(api_?key|token|secret|password)$", re.IGNORECASE)


def redact_secrets(value):
    """Copy of a config with secret-looking values masked, for printing to logs."""
    if isinstance(value, dict):
        out = OrderedDict() if isinstance(value, OrderedDict) else {}
        for k, v in value.items():
            if isinstance(k, str) and _SECRET_KEY_RE.search(k) and isinstance(v, str) and v:
                out[k] = "***"
            else:
                out[k] = redact_secrets(v)
        return out
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    return value


class BaseProcess(object):

    def __init__(
            self,
            process_id: int,
            job: 'BaseJob',
            config: OrderedDict
    ):
        self.process_id = process_id
        self.meta: OrderedDict
        self.job = job
        self.config = config
        self.raw_process_config = config
        self.name = self.get_conf('name', self.job.name)
        self.meta = copy.deepcopy(self.job.meta)
        self.timer: Timer = Timer(f'{self.name} Timer')
        self.performance_log_every = self.get_conf('performance_log_every', 0)

        print(json.dumps(redact_secrets(self.config), indent=4))
        
    def on_error(self, e: Exception):
        pass

    def get_conf(self, key, default=None, required=False, as_type=None):
        # split key by '.' and recursively get the value
        keys = key.split('.')

        # see if it exists in the config
        value = self.config
        for subkey in keys:
            if subkey in value:
                value = value[subkey]
            else:
                value = None
                break

        if value is not None:
            if as_type is not None:
                value = as_type(value)
            return value
        elif required:
            raise ValueError(f'config file error. Missing "config.process[{self.process_id}].{key}" key')
        else:
            if as_type is not None and default is not None:
                return as_type(default)
            return default

    def run(self):
        # implement in child class
        # be sure to call super().run() first incase something is added here
        pass

    def add_meta(self, additional_meta: OrderedDict):
        self.meta.update(additional_meta)


from jobs import BaseJob
