import abc
import os

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers
from collections import OrderedDict
from copy import deepcopy
from torch.func import functional_call, jvp

from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer



class LinearizedModel(nn.Module):
    """Creates a linearized version of a nn.Module.

    The linearized version of a model is a proper PyTorch model and can be
    trained as any other nn.Module.

    Args:
        model (nn.Module): The model to linearize. The trainable parameters of
            the linearized model will be initialized to the parameters of this
            model.
        init_model (nn.Module): A model of the same type as `model` containing
            the parameters around which the model is initialized. If not
            provided, `model` is used as the initialization model.
    """

    def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
        """Initializes the linearized model."""
        super().__init__()
        if init_model is None:
            init_model = model

        func0, params0, self.buffers0 = make_functional_with_buffers(
            init_model.eval(), disable_autograd_tracking=True
        )
        self.func0 = lambda params, x: func0(params, self.buffers0, x)

        _, params, _ = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )

        self.params = nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)
        self._model_name = model.__class__.__name__

        # The intial parameters are not trainable.
        for p in self.params0:
            p.requires_grad = False

        # The params are.
        for p in self.params:
            p.requires_grad = True

    def __call__(self, x) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(
            lambda param: self.func0(param, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp
    
    def dp(self, x) -> torch.Tensor:
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        _, dp = jvp(
            lambda param: self.func0(param, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return dp

def dict_params_to_tuple(dict_params: dict):
    return tuple(v for k, v in dict_params.items())


class LinearizedModelWraper(nn.Module):
    def __init__(self, model: T5ForConditionalGeneration, init_model: T5ForConditionalGeneration = None):
        """
        Initializes a linearized model.

        Args:
            model (nn.Module): The underlying PyTorch model to be linearized.
            init_model (nn.Module): The initial PyTorch model used to compute the linearization parameters (default: None).
        """
        super().__init__()
        self.model = model
        if init_model is None:
            init_model = model
        assert not hasattr(self, "params0")
        params0 = deepcopy([(k, v.detach()) for k, v in init_model.named_parameters()])
        self.params0_keys = [k for k, v in params0]
        self.params0_values = nn.ParameterList([v for k, v in params0])
        for p in self.params0_values:
            p.requires_grad_(False)

    def tuple_params_to_dict(self, tuple_params):
        """
        Converts a tuple of parameters to a dictionary with keys corresponding to the parameter names.

        Args:
            tuple_params (Tuple[Tensor, ...]): A tuple of parameters.

        Returns:
            Dict[str, Tensor]: A dictionary with keys corresponding to the parameter names and values corresponding to the
            parameter values.
        """
        assert len(tuple_params) == len(self.params0_keys)
        state_dict = {}
        for k, p in zip(self.params0_keys, tuple_params):
            state_dict[k] = p
        return state_dict

    def forward(self, *args, **kwargs):
        """
        Computes the linearized model output using a first-order Taylor decomposition.

        Args:
            *args: Positional arguments to be passed to the model.
            **kwargs: Keyword arguments to be passed to the model.

        Returns:
            torch.Tensor: The output of the linearized model, computed using a first-order Taylor decomposition.
        """
        params0 = tuple(self.params0_values)
        params = dict_params_to_tuple(OrderedDict(self.named_parameters()))
        dparams = tuple(p - p0 for p, p0 in zip(params, params0))
        print(dparams)
        out, dp = jvp(
            lambda *param: functional_call(
                self.model, self.tuple_params_to_dict(param), args, kwargs
            ),
            params0,
            dparams,
        )
        return out + dp

class SimpleCallableT5Model(nn.Module):
    def __init__(self, model: T5ForConditionalGeneration):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits
