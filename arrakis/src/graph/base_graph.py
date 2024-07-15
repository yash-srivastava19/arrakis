import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, Any, Union, List, Tuple
from matplotlib.colors import LinearSegmentedColormap

class PlotSpec:
    def __init__(self, plot_type: str, data_keys: Union[str, List[str]], title: str = None, **kwargs):
        self.plot_type = plot_type
        self.data_keys = [data_keys] if isinstance(data_keys, str) else data_keys
        self.title = title
        self.kwargs = kwargs

class MatplotlibWrapper:
        
    def setup(self):
        plt.style.use('seaborn-v0_8-pastel')
        sns.set_style("whitegrid") # whitegrid, darkgrid, dark, white, ticks
        # print(plt.rcParams.keys())
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['font.family'] = 'monospace'
        plt.rcParams['font.monospace'] = ['Roboto Mono']
        # plt.rcParams['font.usetex'] = True
        plt.rcParams['font.size'] = 8
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 6
        plt.rcParams['ytick.labelsize'] = 6
        plt.rcParams['legend.fontsize'] = 6
        plt.rcParams['figure.titlesize'] = 14
        self.colors = ['#FAF3DD','#C8D5B9','#8FC0A9']
        self.cmap = LinearSegmentedColormap.from_list("custom", self.colors, N=256)

    def plot(self, plot_type, cache, data_keys, title, **kwargs):
        self.setup(self) # explicitly we have to do this.
        
        method_name = f"plot_{plot_type}"
        if not hasattr(self, method_name):
            raise ValueError(f"Unknown plot type: {plot_type}")

        return getattr(self, method_name)(self, cache=cache, data_keys=data_keys, title=title, **kwargs)
    
    
    def plot_attention(self, cache, data_keys, title, **kwargs):
        attn_data = cache[data_keys[0]]
        fig, ax = plt.subplots()

        sns.heatmap(attn_data[0], ax=ax, cmap=self.cmap)
        ax.set_title(title or f"Attention Patterns : {data_keys[0]}")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        plt.tight_layout()
        return fig


    def plot_neuron_activation(self, cache, data_keys ,title: str = None, **kwargs) -> plt.Figure:
        # print(cache.keys())
        activations = cache[data_keys[0]].mean(dim=0)
        
        fig, ax = plt.subplots()
        sns.violinplot(data=activations.t(), ax=ax, cmap = self.cmap)
        ax.set_title(title or f"Neuron Activations L {data_keys[0]}")
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Activation")
        plt.tight_layout()

        return fig

    def plot_attention_stats(self, data_keys, title: str = None, **kwargs) -> plt.Figure:
        # attn_data = self._ensure_tensor(data["attention"])
        # path = data.get("path")
        # tokens = data.get("tokens", None)

        # if path is None:
        src_layer, src_head = kwargs.pop("src_layer", 0), kwargs.pop("src_head", 0)
        dst_layer, dst_head = kwargs.pop("dst_layer", 1), kwargs.pop("dst_head", 0)
        path = [(src_layer, src_head, dst_layer, dst_head)]

        # Example matplotlib plotting (replace with your specific plotting logic)
        fig, ax = plt.subplots()
        # Plot attention flow
        ax.text(0.5, 0.5, "Attention Flow", fontsize=12, ha='center')
        ax.axis('off')
        fig.tight_layout()

        return fig

    def plot_induction_heads(self, cache, data_keys, title: str = None, **kwargs) -> plt.Figure:
        attn_patterns = cache[data_keys[0]]
        num_heads = attn_patterns.shape[0]

        fig, axs = plt.subplots(1,num_heads, figsize=(4*num_heads, 4))

        if num_heads > 1:
            for head in range(num_heads):
                sns.heatmap(attn_patterns[head :, ], ax=axs[head], cmap=self.cmap)
                axs[head].set_title(f"Head {head}")
                axs[head].set_xlabel("Key Position")

                if head == 0:
                    axs[head].set_ylabel("Query Position")
                else:
                    axs[head].set_ylabel("")
        else:
            sns.heatmap(attn_patterns[0], ax=axs, cmap=self.cmap)
            axs.set_xlabel("Key Position")
            axs.set_ylabel("Query Position")
            axs.set_title(f"Head 0")
        
        fig.suptitle(title or f"Induction Heads : {data_keys[0]}")
        plt.tight_layout()
        return fig
    

    def plot_neuron_activation_trajectories(self, cache, data_keys, title: str = None, **kwargs) -> plt.Figure:
        activations = cache[data_keys[0]]
        neuron_magnitudes = activations.abs().mean(dim=0)
        top_k_neurons = torch.topk(neuron_magnitudes, k=5).indices

        fig, ax = plt.subplots()
        for neuron in top_k_neurons:
            ax.plot(activations[:, neuron], label=f"Neuron {neuron}")
        
        ax.set_title(title or f"Neuron Activation Trajectories : {data_keys[0]}")
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Activation")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig

    def plot_qk_dot_product(self, cache, data_keys, title, **kwargs):
        # print(cache.keys())

        q = cache[data_keys[0]]
        k = cache[data_keys[1]]
        
        qk = q @ k.transpose(-1, -2)

        fig, ax = plt.subplots()
        sns.heatmap(qk, ax=ax, cmap=self.cmap, center=0)
        ax.set_title(title or "Average QK Dot Products")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        plt.tight_layout()

        return fig


    def plot_residual_stream(self, cache, data_keys, title: str = None, **kwargs) -> plt.Figure:
        res_keys = sorted([k for k in cache.keys() if "hook_resid_post" in k])
        res_norms = [torch.norm(cache[k]).item() for k in res_keys]

        fig, ax = plt.subplots()
        fig.suptitle(title or "Residual Stream Norm Progression")

        sns.lineplot(x=range(len(res_keys)), y=res_norms, ax=ax, markers='o')
        
        ax.set_xticks(range(len(res_keys)))
        ax.set_xticklabels([f"L{i}" for i in range(len(res_keys))], ha='right')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Residual Norm")
        
        plt.tight_layout()
        return fig
    
    def plot_layer_norms(self, cache, data_keys, title, **kwargs):
        ln_keys = sorted([k for k in cache.keys() if "ln" in k]) # have to see how to use model_attrs here.
        ln_stats = [torch.norm(cache[k]).item() for k in ln_keys]

        fig, ax = plt.subplots()
        ax.plot(ln_stats, marker='o')
        ax.set_xticks(range(len(ln_keys)))
        ax.set_xticklabels(ln_keys, rotation=45, ha='right')
        ax.set_title(title or "Layer Norm Statistics")
        ax.set_ylabel("Norm")
        plt.tight_layout()

        return fig