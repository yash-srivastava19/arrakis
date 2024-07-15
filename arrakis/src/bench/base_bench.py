import json
import time
import torch
import statistics as stats  # see if this import is there or something else.
import hashlib
import inspect
from datetime import datetime
from collections import defaultdict
from functools import wraps
from arrakis.src.core_arrakis.attention_tools import AttentionTools
from arrakis.src.core_arrakis.logit_attribution import LogitAttributor
from arrakis.src.core_arrakis.knowledge_prober import KnowledgeProber
from arrakis.src.core_arrakis.causal_tracing_intervention import CausalTracer
from arrakis.src.core_arrakis.superposition_disentangler import SuperpositionDisentangler
from arrakis.src.core_arrakis.sparsity_analyzer import SparsityAnalyzer
from arrakis.src.core_arrakis.read_write_heads import WriteReadAnalyzer
from arrakis.src.core_arrakis.attention_head_comp import AttentionComposer
from arrakis.src.core_arrakis.kg_extractor import KnowledgeGraphExtractor
from arrakis.src.core_arrakis.logit_lens import LogitLens
from arrakis.src.core_arrakis.residual_decomposer import ResidualDecomposer
from arrakis.src.core_arrakis.residual_tools import ResidualTools
from arrakis.src.core_arrakis.knowledge_prober import KnowledgeProber
from arrakis.src.core_arrakis.model_surgery import ModelSurgery
from arrakis.src.core_arrakis.feature_viz import FeatureViz
"""Base Interpretablity Bench. Keep in mind we are not dealing with tokenization and everything. We need the ids, an we'll work with that."""
class BaseInterpretabilityBench:
    def __init__(self, model, save_dir="experiments"):
        self.model = model
        self.save_dir = save_dir
        self.tools = {
            "attention": AttentionTools(model), 
            "composer": AttentionComposer(model), 
            "logit": LogitAttributor(model), 
            "logit_lens": LogitLens(model),
            "knowledge": KnowledgeProber(model), 
            "kg_extractor": KnowledgeGraphExtractor(model),
            "causal": CausalTracer(model), 
            "decomposer": ResidualDecomposer(model),
            "superposition": SuperpositionDisentangler(model), 
            "sparsity": SparsityAnalyzer(model), 
            "write_read": WriteReadAnalyzer(model),
            "residual": ResidualTools(model),
            "k_prober": KnowledgeProber(model),
            "surgery": ModelSurgery(model),
            "feature_viz": FeatureViz(model)

            # Add more tools here
        }

        self._experiment_versions = {}
        self._experiments = {}
        self._plotting_lib = None

    def set_plotting_lib(self, lib):
        self._plotting_lib = lib

    def plot_results(self, *plots, input_ids, **kwargs):
        def decorator(func) :
            @wraps(func)
            def wrapper(*args, **kwargs):
                if self._plotting_lib is None:
                    raise ValueError("Plotting library not set. Use set_plotting_lib() first.")
                
                figs = []
                _, cache = self.model(input_ids)
                for plot in plots:
                    fig = self._plotting_lib.plot(
                        self._plotting_lib,
                        cache = cache,
                        plot_type=plot.plot_type,
                        data_keys=plot.data_keys,
                        title=plot.title,
                        **kwargs
                    )
                    figs.append(fig)
                
                return figs
            return wrapper
        return decorator


    def log_experiment(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            experiment_name = func.__name__
            print(f"Running experiment: {experiment_name}")
            result = func(*args, **kwargs)
            # print("Result: ", result)
            exp_data = {
                "args": args,
                "kwargs": kwargs,
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "source": inspect.getsource(func)
            }

            # print(exp_data)

            hasher = hashlib.sha256()
            hasher.update(json.dumps(exp_data, default=self.json_encoder).encode())
            version_hash = hasher.hexdigest()[:8]  # First 8 characters of the hash
            
            if func.__name__ not in self._experiment_versions:
                self._experiment_versions[func.__name__] = []
            self._experiment_versions[func.__name__].append(version_hash)
            
            self._experiments[f"{func.__name__}_v{version_hash}"] = exp_data

            # Save results
            save_path = f"{self.save_dir}/{experiment_name}.json"
            with open(save_path, "w") as f:
                json.dump(exp_data, f, indent=4, default=self.json_encoder)
            
            print(f"Results saved to {save_path}")
            return result
        return wrapper

    def list_versions(self, experiment_name):
        return self._experiment_versions.get(experiment_name, [])
    
    def get_version(self, experiment_name, version_hash):
        key = f"{experiment_name}_v{version_hash}"
        return self._experiments.get(key, None)
    
    def test_hypothesis(self, null_func, alt_func, n_samples=100, alpha=0.05):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)

                # Generate samples under null and alternate hypothesis
                null_samples = [null_func(*args, **kwargs) for _ in range(n_samples)]
                alt_samples = [alt_func(*args, **kwargs) for _ in range(n_samples)]

                t_stat, p_value = stats.ttest_ind(null_samples, alt_samples)

                result["hypothesis_test"] = {
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "reject_null": p_value < alpha
                }
                return result 
            return wrapper
        return decorator
    
    def profile_model(self, granularity="layer"):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                model = self.model
                timings = defaultdict(float)
                activation_sizes = defaultdict(list)

                def layer_hook(name):
                    def hook(module, input, output):
                        t0 = time.time()
                        yield 
                        timings[name] += time.time() - t0
                        activation_sizes[name].append(output.numel() * output.element_size())
                    return hook
                
                hooks = []
                for name, module in model.named_modules():
                    if granularity == "layer" and "layers." in name:
                        hooks.append(module.register_forward_hook(layer_hook(name), prepend=True))
                    elif granularity == "component" and any(x in name for x in ["attention", "mlp", "layer_norm"]):
                        hooks.append(module.register_forward_hook(layer_hook(name), prepend=True))

                result = func(model, *args[1:], **kwargs)

                for hook in hooks:
                    hook.remove()

                total_time = sum(timings.values())
                total_memory = sum(max(sizes) for sizes in activation_sizes.values())

                result["profiling"] = {
                    "time_percentages": {k: v / total_time for k, v in timings.items()},
                    "memory_percentages": {k: max(v) / total_memory for k, v in activation_sizes.items()},
                    "total_time": total_time,
                    "total_memory": total_memory
                }
                return result
            return wrapper
        return decorator
    
    def use_tools(self, *tools):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                required_tools = [self.tools[tool] for tool in tools]
                return func(*args, **kwargs, **{tool: tool_instance for tool, tool_instance in zip(tools, required_tools)})
            return wrapper
        return decorator


    @staticmethod
    def json_encoder(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return str(obj)