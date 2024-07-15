from transformers import PreTrainedModel, PretrainedConfig
from transformers import GPT2Model, GPTNeoModel, GPTNeoXModel, LlamaModel, GemmaModel, Phi3Model, T5Model, Qwen2Model, BloomModel, MistralModel, StableLmModel, MixtralModel
from transformers import GPT2Config, GPTNeoConfig, GPTNeoXConfig, LlamaConfig, GemmaConfig, Phi3Config, T5Config, Qwen2Config, BloomConfig, MistralConfig, StableLmConfig, MixtralConfig

import torch 
import torch.nn as nn
import logging
import pathlib 
"""Helper class to register hooks on a model and get activations."""

#TODO: Can't seem to figure out how there is an empty entry in the activations.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, filename=pathlib.Path.joinpath(pathlib.Path.cwd(), "logs", "arrakis_model.log"), filemode="w", format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

H_MODELS = ["gpt2", "gpt-neo"]

LAYER_MODELS = ["llama", "gemma", "phi3", "t5", "qwen2", "mistral", "stable-lm", "gpt-neox"]

DIFFERENT_KINDA = ["bloom"]

MODELS_THAT_ARE_SUPPORTED = [
    "gpt2",
    "gpt-neo",
    "gpt-neox",
    "llama",
    "gemma",
    "phi3",
    "qwen2",
    "mistral",
    "stable-lm",

]

MODEL_CLASS_MAPPING = {
    "gpt2": GPT2Model,
    "gpt-neo": GPTNeoModel,
    "gpt-neox": GPTNeoXModel,
    "llama": LlamaModel,
    "gemma": GemmaModel,
    "phi3": Phi3Model,
    "t5": T5Model,
    "qwen2": Qwen2Model,
    "mistral": MistralModel,
    "stable-lm": StableLmModel,
}

MODEL_CONFIG_MAPPING = {
    "gpt2": GPT2Config,
    "gpt-neo": GPTNeoConfig,
    "gpt-neox": GPTNeoXConfig,
    "llama": LlamaConfig,
    "gemma": GemmaConfig,
    "phi3": Phi3Config, 
    "t5": T5Config,
    "qwen2": Qwen2Config,
    "mistral": MistralConfig,
    "stable-lm": StableLmConfig,

}

class ActivationCache:
    """Registers hooks on a model and stores the activations."""
    def __init__(self, model):
        self.model = model
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self):
        """Registers hooks on the model."""
        def hook_fn(name, is_forward=True):
            def register_hook(module, input, output=None):
                # Only works with register_forward_hook(needs output), not pre_hook.
                if is_forward:
                    self.activations[name] = output[0].detach().cpu()
                else:
                    # See this hacky fix. It is working, but idk why. I'll look into it later.
                    if len(input) >= 1:
                        if type(input) == tuple:
                            # print(input)
                            self.activations[name] = input[0].detach().cpu()
                    else:
                        self.activations[name] = torch.tensor(input).detach().cpu()
            return register_hook

        logger.debug(f"Registering hooks for {self.model.name}")
        for name, submodule in self.model.model.named_modules():
            if hasattr(submodule, "register_forward_hook"):
                if name=="": # This is the empty entry that I was talking about. It is the whole model.
                    self.hooks.append(submodule.register_forward_hook(hook_fn("root")))
                else:
                    self.hooks.append(submodule.register_forward_hook(hook_fn(name)))

        # Now, we can just update this as model_attrs takes care for us :)
        logger.debug(f"Registering iterative hooks(or blocks/layers) for {self.model.name}")
        for i, block in enumerate(self.model.model_attrs.get_block()):
            
            block_type = self.model.model_attrs.get_block_type()
            attn_type = self.model.model_attrs.get_attn_type()
            # The ones starting with block are backward hooks. The ones starting with blocks are forward hooks.
            block.register_forward_pre_hook(hook_fn(f"{block_type}.{i}.hook_resid_pre", is_forward=False)) # This line breaks. Fixed
            block.register_forward_hook(hook_fn(f"{block_type}.{i}.hook_resid_post"))
            
            mlp = self.model.model_attrs.get_mlp(i)
            mlp.register_forward_hook(hook_fn(f"{block_type}.{i}.mlp.hook_result_pre", is_forward=False))
            mlp.register_forward_hook(hook_fn(f"{block_type}.{i}.mlp.hook_result_post"))

            
            attn = self.model.model_attrs.get_attn(i)
            attn.register_forward_hook(hook_fn(f"{block_type}.{i}.{attn_type}.hook_result_pre",is_forward=False))
            attn.register_forward_hook(hook_fn(f"{block_type}.{i}.{attn_type}.hook_result_post"))
            
            q = self.model.model_attrs.get_q(i) ; k = self.model.model_attrs.get_k(i) ; v = self.model.model_attrs.get_v(i)
            q_type = self.model.model_attrs.get_q_type() ; k_type = self.model.model_attrs.get_k_type() ; v_type = self.model.model_attrs.get_v_type()
            
            q.register_forward_hook(hook_fn(f"{block_type}.{i}.{attn_type}.{q_type}.hook_result_pre", is_forward=False))
            q.register_forward_hook(hook_fn(f"{block_type}.{i}.{attn_type}.{q_type}.hook_result_post"))

            k.register_forward_hook(hook_fn(f"{block_type}.{i}.{attn_type}.{k_type}.hook_result_pre", is_forward=False))
            k.register_forward_hook(hook_fn(f"{block_type}.{i}.{attn_type}.{k_type}.hook_result_post"))

            v.register_forward_hook(hook_fn(f"{block_type}.{i}.{attn_type}.{v_type}.hook_result_pre", is_forward=False))
            v.register_forward_hook(hook_fn(f"{block_type}.{i}.{attn_type}.{v_type}.hook_result_post"))
            
            

        
        logger.debug(f"Registering hooks for the last module(logits) of the model : {self.model.name}")
        last_module_name = list(self.model.named_modules())[-1][0]
        last_module = dict(self.model.named_modules())[last_module_name]
        last_module.register_forward_hook(hook_fn("logits"))      

def register_hooks(forward_fn):
    """Decorator to register hooks on the model."""
    logger.debug(f"Invoked Activation Cache. Registering hooks for : {forward_fn.__name__}")
    def wrapped_forward(self, *args, **kwargs):
        cache = ActivationCache(self)
        cache.register_hooks()

        output = forward_fn(self, *args, **kwargs)
        activations = cache.activations

        return output, activations
    
    return wrapped_forward

class ModelAttributes:
    """Helper class to get the attributes of the model, their names and attributes."""
    def __init__(self, model):
        self.model = model
        
        # These attributes are used by us in the main library.
        self.block_type = None
        self.attn_type = None
        self.q_type = None
        self.k_type = None
        self.v_type = None
        self.o_type = None
        self.mlp_in_type = None
        self.mlp_out_type = None
        self.lin_ff_type = None # Yes. Used in kg-prober.
        self.embed_type = None # Yes. Used in 


        self._models_exceptions_list = ["get-neox", "phi3", "bloom"]
        self._qkv_exceptions_list = []
        self._o_exceptions_list = []

        self.setup()
    
    def setup(self):
        self.model_str = repr(self.model)
        self.parse(self.model_str)

    def parse(self, model_str):
        """Parses the model string to get the attributes of the model. There are better ways to do this :) """
        # Update; There must be a clever way to do this using ast. I'll look into it.
        # We'll start from the embeddings and go till the output layer.
        # print(model_str)
        if "(wte):" in model_str:
            self.embed_type = "wte"
        elif "(word_embeddings):" in model_str:
            self.embed_type = "word_embeddings"
        elif "(embed_tokens):" in model_str:
            self.embed_type = "embed_tokens"
        elif "(embed_in):" in model_str:
            self.embed_type = "embed_in"
        else:
            self.embed_type = "<-break->"
        
        # First, let's see what are the main blocks called.
        if "(h):" in model_str:
            self.block_type = "h"
        elif "(blocks):" in model_str:
            self.block_type = "blocks"
        elif "(layers):" in model_str:
            self.block_type = "layers"
        else:
            self.block_type = "<-break->"

        block_module = model_str[model_str.index(self.block_type):]
        # print(block_module)
        # Now, let's see what are the main attention blocks called.
        if "(attn):" in block_module:
            self.attn_type = "attn"
        elif "(attention):" in block_module:
            self.attn_type = "attention"
        elif "(self_attn):" in block_module:
            self.attn_type = "self_attn"
        elif "(self_attention):" in block_module:
            self.attn_type = "self_attention"
        else:
            self.attn_type = "<-break->" # attention is now accessed by model.block_type.attn_type

        # Inside attn_type, let's see what are the main attention components called.
        # print(model_str)

        # This is the troublemaker. Here parsing is not accurate.
        attn_module = model_str[model_str.index(self.attn_type):]
        # print(attn_module) 

        if "(c_attn):" in attn_module and "(c_proj):" in attn_module:
            self.q_type = "c_attn" ; self.k_type = "c_attn" ; self.v_type = "c_attn" ; 
            if "(c_proj):" in attn_module:
                self.o_type = "c_proj"

        elif "(q_proj):" in attn_module and "(k_proj):" in attn_module and "(v_proj):" in attn_module:
            self.q_type = "q_proj" ; self.k_type = "k_proj" ; self.v_type = "v_proj"
            if "(out_proj):" in attn_module:
                self.o_type = "out_proj"
            elif "(o_proj):" in attn_module:
                self.o_type = "o_proj"
            else:
                self.o_type = "<-break->"

        else:
            # If we are reaching here, then it means that q_k_v are all bundled. Currently, this happens for get-neox, phi3, and bloom. All are dealt with here.
            if "(query_key_value):" in attn_module:
                self.q_type = "query_key_value"
                self.k_type = "query_key_value"
                self.v_type = "query_key_value"
                if "dense" in attn_module:
                    self.o_type = "dense"
            
            elif "(qkv_proj):" in attn_module:
                self.q_type = "qkv_proj"
                self.k_type = "qkv_proj"
                self.v_type = "qkv_proj"
                if "o_proj" in attn_module:
                    self.o_type = "o_proj"
        
        mlp_module = model_str[model_str.index(self.o_type):]
        # print(mlp_module)
        if "(c_fc):" in mlp_module and "(c_proj):" in mlp_module: # Gpt, GPTNeo
            self.mlp_in_type = "c_fc" ; self.mlp_out_type = "c_proj"
        
        elif "(dense_h_to_4h):" in mlp_module and "(dense_4h_to_h)" in mlp_module:
            self.mlp_in_type = "dense_h_to_4h" ; self.mlp_out_type = "dense_4h_to_h"

        elif "(gate_proj):" in mlp_module and "(down_proj):" in mlp_module: # Gemma, Llama
            self.mlp_in_type = "gate_proj" ; self.mlp_out_type = "down_proj"

        elif "(gate_up_proj):" in mlp_module and "(down_proj):" in mlp_module: # Phi3
            self.mlp_in_type = "gate_up_proj" ; self.mlp_out_type = "down_proj"
        
        elif "(fc_in):" in mlp_module and "(fc_out):" in mlp_module: # Qwen2
            self.mlp_in_type = "fc_in" ; self.mlp_out_type = "fc_out"
        
        lin_ff_module = model_str[model_str.index(self.mlp_out_type):]
        if "(ln_f):" in lin_ff_module:
            self.lin_ff_type = "ln_f"
        elif "(final_layer_norm)" in lin_ff_module:
            self.lin_ff_type = "final_layer_norm"
        elif "(norm):" in lin_ff_module:
            self.lin_ff_type = "norm"
        else:
            self.lin_ff_type = "<-break->"
        # This parser seems to be working with all except get-neox, phi3, and bloom. one reason is that they have combined q_k_v in one layer. output name is also different.
        # Upate: I've added cases that we can cover till now :) We'll deal with the rest later.

    def get_block_type(self):
        """Returns the block type of the model."""
        return self.block_type
    

    def get_attn_type(self):
        """Returns the attention type of the model."""
        return self.attn_type
    
    def get_q_type(self):
        """Returns the query type of the model."""
        return self.q_type
    
    def get_k_type(self):
        """Returns the key type of the model."""
        return self.k_type
    
    def get_v_type(self):
        """Returns the value type of the model."""
        return self.v_type
    
    def get_o_type(self):
        """Returns the output type of the model."""
        return self.o_type
    
    def get_mlp_in_type(self):
        """Returns the input type of the MLP layer."""
        return self.mlp_in_type
    
    def get_mlp_out_type(self):
        """Returns the output type of the MLP layer."""
        return self.mlp_out_type
    
    def get_lin_ff_type(self):
        """Returns the final layer norm type of the model."""
        return self.lin_ff_type
    
    def get_embed_type(self):
        """Returns the embedding type of the model."""
        return self.embed_type
    
    def get_embed(self):
        """Returns the embedding layer of the model."""
        return getattr(self.model, self.embed_type)
    
    def get_lin_ff(self):
        """Returns the final layer norm of the model."""
        return getattr(self.model, self.lin_ff_type)
    
    def get_block(self, layer_idx=None):
        """Returns the block of the model. If layer_idx is provided, returns the block at that index."""
        if layer_idx:
            return getattr(self.model, self.block_type)[layer_idx]  # This might change if we want to make it an attribute inside `HookedModel`
        return getattr(self.model, self.block_type)
    
    def set_block(self, block):
        """Sets the block of the model. Used in model_surgery"""
        setattr(self.model, self.block_type, block)

    
    def get_attn(self, layer_idx):
        """Returns the attention layer of the model."""
        if isinstance(self.get_block(layer_idx), nn.ModuleList):
            return getattr(self.get_block(layer_idx)[0], self.attn_type)
        
        return getattr(self.get_block(layer_idx), self.attn_type) # could be anything.

    def get_mlp(self, layer_idx):
        """Returns the MLP layer of the model."""
        if isinstance(self.get_block(layer_idx), nn.ModuleList):
            return getattr(self.get_block(layer_idx)[0], "mlp")
        return getattr(self.get_block(layer_idx), "mlp")

    def get_mlp_in(self, layer_idx):
        """Returns the input layer of the MLP layer."""
        return getattr(self.get_mlp(layer_idx), self.mlp_in_type)
    
    def get_mlp_out(self, layer_idx):
        """Returns the output layer of the MLP layer."""
        return getattr(self.get_mlp(layer_idx), self.mlp_out_type)

    def get_q(self, layer_idx):
        """Returns the query layer of the attention layer."""
        # If the model is GPTNeo, there is an extra layer of wrapping. GPTNeoAttention -> GPTNeoSelfAttention
        if "GPTNeoModel" in self.model_str:
            return getattr(self.get_attn(layer_idx).attention, self.q_type)
        return getattr(self.get_attn(layer_idx), self.q_type)
    
    def get_k(self, layer_idx):
        """Returns the key layer of the attention layer."""
        if "GPTNeoModel" in self.model_str:
            return getattr(self.get_attn(layer_idx).attention, self.k_type)
        return getattr(self.get_attn(layer_idx), self.k_type)

    def get_v(self, layer_idx):
        """Returns the value layer of the attention layer."""
        if "GPTNeoModel" in self.model_str:
            return getattr(self.get_attn(layer_idx).attention, self.v_type)
        return getattr(self.get_attn(layer_idx), self.v_type)
     
    def get_o(self, layer_idx):
        """Returns the output layer of the attention layer."""
        if "GPTNeoModel" in self.model_str:
            return getattr(self.get_attn(layer_idx).attention, self.o_type)
        return getattr(self.get_attn(layer_idx), self.o_type)
    
    def __repr__(self):
        """Returns the string representation of the model attributes."""
        return f"ModelAttributes(block_type={self.block_type}, attn_type={self.attn_type}, q_type={self.q_type}, k_type={self.k_type}, v_type={self.v_type}, o_type={self.o_type}, mlp_in_type={self.mlp_in_type}, mlp_out_type={self.mlp_out_type})"



class HookedAutoConfig(PretrainedConfig):
    """Config class for the models. This is a wrapper around the HF Pretrained config class."""
    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug(f"Invoked HookedAutoConfig. Creating config for {name}. Args are {args} and kwargs are {kwargs}")
        if name not in MODELS_THAT_ARE_SUPPORTED:
            raise ValueError(f"{name} is not supported. Supported models are {MODELS_THAT_ARE_SUPPORTED}")
        if name not in MODEL_CONFIG_MAPPING:
            raise ValueError(f"{name} is not supported. Supported models are {MODEL_CONFIG_MAPPING.keys()}")
        
        self.name = name
        self.config  = MODEL_CONFIG_MAPPING[name](*args, **kwargs)


class HookedAutoModel(PreTrainedModel):
    """Model class for the models. This is a wrapper around the HF Pretrained model class."""
    def __init__(self, config:HookedAutoConfig, *args, **kwargs):
        super().__init__(config)
        logger.debug(f"Invoked HookedAutoModel. Creating model for {config.name}")
        self.name = config.name  # Already tested what all models are supported.
        self.model = MODEL_CLASS_MAPPING[self.name](config.config, *args, **kwargs)
        self.model_attrs = ModelAttributes(self.model)

    @register_hooks
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)