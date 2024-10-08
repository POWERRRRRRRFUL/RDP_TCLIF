from abc import abstractmethod
from typing import Callable
import torch
from spiking_neuron import base


class BaseNode(base.MemoryModule):
    def __init__(self,
                 v_threshold: float = 1.,
                 v_reset: float = 0.,
                 surrogate_function: Callable = None,
                 detach_reset: bool = False,
                 step_mode='s', backend='torch',
                 store_v_seq: bool = False):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold

        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq


    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset

        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):
        # 确保 v1 和 v2 在第一次使用时初始化为与 x 大小一致的张量
        self.v_float_to_tensor(x)

        # 动态初始化 v1 和 v2 为与 x 形状一致的张量
        if self.names['v1'].dim() == 0 or self.names['v1'].size() != x.size():
            self.names['v1'] = torch.zeros_like(x)
        if self.names['v2'].dim() == 0 or self.names['v2'].size() != x.size():
            self.names['v2'] = torch.zeros_like(x)

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class TCLIFNode(BaseNode):
    def __init__(self,
                 v_threshold=1.,
                 v_reset=0.,
                 surrogate_function: Callable = None,
                 detach_reset=False,
                 hard_reset=False,
                 step_mode='s',
                 k=2,
                 decay_factor: torch.Tensor = torch.full([1, 2], 0, dtype=torch.float),
                 gamma: float = 0.5):
        super(TCLIFNode, self).__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode)
        self.k = k
        for i in range(1, self.k + 1):
            self.register_memory('v' + str(i), torch.tensor(0.))  # 将 v1 和 v2 初始化为张量

        self.names = self._memories
        self.hard_reset = hard_reset
        self.gamma = gamma
        self.decay = decay_factor
        self.decay_factor = torch.nn.Parameter(decay_factor)

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        # 动态初始化 v1 和 v2 为与 x 形状一致的张量
        if self.names['v1'].dim() == 0:
            self.names['v1'] = torch.zeros_like(x)
        if self.names['v2'].dim() == 0:
            self.names['v2'] = torch.zeros_like(x)

        # 执行张量运算
        # v1: membrane potential of dendritic compartment
        # v2: membrane potential of somatic compartment
        self.names['v1'] = self.names['v1'] - torch.sigmoid(self.decay_factor[0][0]) * self.names['v2'] + x
        self.names['v2'] = self.names['v2'] + torch.sigmoid(self.decay_factor[0][1]) * self.names['v1']
        self.v = self.names['v2']

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if not self.hard_reset:
            # soft reset
            self.names['v1'] = self.jit_soft_reset(self.names['v1'], spike_d, self.gamma)
            self.names['v2'] = self.jit_soft_reset(self.names['v2'], spike_d, self.v_threshold)
        else:
            # hard reset
            for i in range(2, self.k + 1):
                self.names['v' + str(i)] = self.jit_hard_reset(self.names['v' + str(i)], spike_d,  self.v_reset)

    def forward(self, x: torch.Tensor):
        return super().single_step_forward(x)

    def extra_repr(self):
        return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, " \
               f"hard_reset={self.hard_reset}, " \
               f"gamma={self.gamma}, k={self.k}, step_mode={self.step_mode}, backend={self.backend}"

