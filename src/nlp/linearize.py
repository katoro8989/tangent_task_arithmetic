import abc
import os

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers
from collections import OrderedDict
from copy import deepcopy
from torch.func import functional_call, jvp

from transformers import Trainer, TrainingArguments, T5ForConditionalGeneration, T5Tokenizer, PreTrainedModel, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithPast



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


class LinearizedPreTrainedModel(PreTrainedModel):
    def __init__(self, config, original_model, params0_values):
        super().__init__(config)
        self.original_model = original_model
        self.params0_values = params0_values

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
        params0 = tuple(self.params0_values)
        params = dict_params_to_tuple(OrderedDict(self.named_parameters()))
        dparams = tuple(p - p0 for p, p0 in zip(params, params0))
        out, dp = jvp(
            lambda *param: functional_call(
                self.original_model, self.tuple_params_to_dict(param), args, kwargs
            ),
            params0,
            dparams,
        )
        return out + dp
    
    def dp(self, *args, **kwargs):

        params0 = tuple(self.params0_values)
        params = dict_params_to_tuple(OrderedDict(self.named_parameters()))
        dparams = tuple(p - p0 for p, p0 in zip(params, params0))
        _, dp = jvp(
            lambda *param: functional_call(
                self.original_model, self.tuple_params_to_dict(param), args, kwargs
            ),
            params0,
            dparams,
        )
        return dp

class LinearizedGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config, original_model, params0_values, params0_keys):
        super().__init__(config)
        self.original_model = original_model
        self.params0_values = params0_values
        self.params0_keys = params0_keys
        
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
        params0 = tuple(self.params0_values)
        params = dict_params_to_tuple(OrderedDict(self.named_parameters()))
        dparams = tuple(p - p0 for p, p0 in zip(params, params0))
        out, dp = jvp(
            lambda *param: functional_call(
                self.original_model, self.tuple_params_to_dict(param), args, kwargs
            ),
            params0,
            dparams,
        )
        return out + dp
        # return CausalLMOutputWithPast(logits=out + dp)
    
    def dp(self, *args, **kwargs):

        params0 = tuple(self.params0_values)
        params = dict_params_to_tuple(OrderedDict(self.named_parameters()))
        dparams = tuple(p - p0 for p, p0 in zip(params, params0))
        _, dp = jvp(
            lambda *param: functional_call(
                self.original_model, self.tuple_params_to_dict(param), args, kwargs
            ),
            params0,
            dparams,
        )
        return dp
        
    def generate(self, **kwargs):
        if "input_ids" not in kwargs and "inputs" not in kwargs:
            raise ValueError("`input_ids` must be provided for generation.")
        if "input_ids" in kwargs:
            kwargs["inputs"] = kwargs.pop("input_ids")
        return super().generate(**kwargs)

class LinearizedT5Wrapper(nn.Module):
    def __init__(self, model: PreTrainedModel, init_model: PreTrainedModel = None):
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
        self.linearized_model = LinearizedPreTrainedModel(
            model.model.config, model.model, self.params0_values
        )

    def forward(self, *args, **kwargs):
        return self.linearized_model(*args, **kwargs)
    
    def dp(self, *args, **kwargs):
        decoder_input_idsがない場合は自動的に生成
        if 'decoder_input_ids' not in kwargs:
            batch_size = kwargs['input_ids'].size(0)
            kwargs['decoder_input_ids'] = torch.zeros((batch_size, 1), dtype=torch.long, device=kwargs['input_ids'].device)

        return self.linearized_model.dp(*args, **kwargs)
    
    def generate(self, **kwargs):
        return self.linearized_model.generate(**kwargs)

class LinearizedGPT2Wrapper(nn.Module):
    def __init__(self, model: PreTrainedModel, init_model: PreTrainedModel = None):
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

        self.linearized_model = LinearizedGPT2LMHeadModel(
            model.model.config, model, self.params0_values, self.params0_keys
        )

    def forward(self, *args, **kwargs):
        return self.linearized_model(*args, **kwargs)
    
    def dp(self, *args, **kwargs):
        
        return self.linearized_model.dp(*args, **kwargs)
    
    def generate(self, **kwargs):
        return self.linearized_model.generate(**kwargs)

    

class SimpleCallableHFModel(nn.Module):
    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model
    
    def forward(self, *args, **kwargs):
        # return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, decoder_input_ids=decoder_input_ids).logits
        return self.model(*args, **kwargs).logits

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
    
    def generate(self, *args, **kwargs):
        """
        Generates sequences using the underlying `self.model`.
        Passes all arguments directly to the underlying model's `generate` method.
        """
        return self.model.generate(*args, **kwargs)