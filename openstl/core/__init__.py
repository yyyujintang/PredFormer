from .ema_hook import EMAHook, SwitchEMAHook
from .hooks import Hook, Priority, get_priority
from .metrics import metric
from .recorder import Recorder
from .optim_scheduler import get_optim_scheduler
from .optim_constant import optim_parameters
from .drop_scheduler import drop_scheduler

hook_maps = {
    'emahook': EMAHook,
    **dict.fromkeys(['semahook', 'switchemahook'], SwitchEMAHook),
}

__all__ = [
    'Hook', 'EMAHook', 'SwitchEMAHook', 'Priority', 'get_priority', 'metric',
    'Recorder', 'get_optim_scheduler', 'optim_parameters', 'drop_scheduler'
]