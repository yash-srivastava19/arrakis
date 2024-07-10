# Arrakis - A Mechanistic Interpretability Tool

Interpretability is a relatively new field where everyday something new is happening. Mechanistic Interpretbaility is one of the approach to reverse engineer neural networks and understand what is happening inside these black-box models.

Mechanistic Interpretability is a really exciting subfield of alignment, and recently, a lot has been happening in this field - especially at Anthropic. To look at the goal of MI at Anthropic, read this [post](https://www.lesswrong.com/posts/CzZ6Fch4JSpwCpu6C/interpretability). The core operation involved in MI is loading a model, looking at it's weights and activations, and doing some operations on them and producing results. 

I made **Arrakis** to deeply understand Transformer based models(maybe in future I will try to be model agnostic). The first thought that should come to mind is *Why not use Transformer Lens? Neel Nanda has already made significant progress in that*. I made Arrakis as I wanted to have a library which can do more than just get the activations - I wanted a more complete library where researchers can do experiments, and track their progress. Think of **Arrakis** as a complete suite to conduct MI experiments, where I try to get the best of both [Transformer Lens](https://transformerlensorg.github.io/TransformerLens/) and [Garcon](https://transformer-circuits.pub/2021/garcon/index.html). More features will be added as I understand how to made this library more useful for the community, and I need feedback for that.

## Tools and Decomposibility

Regardless of what research project you are working on, if you are not keeping track of things, it gets messy really easily. In a field like MI, where you are constantly looking at all different weights and biases, and there are a lot of moving parts - it gets overwhelming fairly easily. I've experienced this personally, and being someone who is obsessed with reducing experimentation time and get results quickly, I wanted to have a complete suite which makes my workload easy. 

Arrakis is made so that this doesn't happen. The core principle behind Arrakis is decomposibility. Do all experiments with plug-and-play tools(will be much clear in the walkthrough). This makes experimentation really flexible, and at the same time, Arrakis keeps track of different versions of the experiments by default. Everything in Arrakis is made in this plug and play fashion. I have even incorporated a graphing library(on top of several popular libraries) to make graphing a lot easier.

I really want feedback and contributions on this project so that this can be adapted by the community at large.

## Arrakis Walkthrough
Let's understand how to conduct a small experiment in Arrakis. It is easy, reprodcible and a lot easy to implement.

### Step 1: Install the package
All the dependencies of the project are maintained through poetry.
```python
pip install arrakis  # TODO: In future.
```

### Step 2: Create HookedAutoModel
`HookedAutoModel` offers a convinient way to import models from Huggingface directly(with Hooks). Everything just works out of the box. First, create a HookedConfig for the model you want to support with the required parameters. Then, create a `HookedAutoModel` from the config. As of now, these models are supported : 

```python
[ 
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
```

As mentioned, the core idea behind Arrkis is decompsibility, so a `HookedAutoModel` is a wrapper around Huggingface `PreTrainedModel` class, with a single plug and play decorator for the forward pass. All the model probing happens behind the scenes, and is pre-configured.

```python
from arraki.src.core_arrakis.activation_cache import *

config = HookedAutoConfig(name="llama", 
    vocab_size=50256, 
    hidden_size=8, 
    intermediate_size=2, 
    num_hidden_layers=4, 
    num_attention_heads=4,
    num_key_value_heads=4)

model = HookedAutoModel(config)

```

### Step 3: Set up Interpretability Bench

At it's core, the whole purpose of Arrakis is to conduct MI experiment. After installing, derive from the `BaseInterpretabilityBench` and instantiate an object(`exp` in this case). This object provides a lot of function out-of the box based on the "tool" you want to use for the experiment, and have access to the functions that the tool provides. You can also create your own tool(read about that [here](README.md#extending-arrakis) )

```python
from arrakis.src.core_arrakis.base_bench import BaseInterpretabilityBench

class MIExperiment(BaseInterpretabilityBench):
    def __init__(self, model, save_dir="experiments"):
        super().__init__(model, save_dir)
        self.tools.update({"custom": CustomFunction(model)})

exp = MIExperiment(model)
```

Apart from access to MI tools, the object also provies you a convinient way to log your experiments. To log your experiments, just decorate the function you are working with `@exp.log_experiment`, and that is pretty much it. The function creates a local version control on the contents of the function, and stores it locally. You can run many things in parallel, and the version control helps you keep track of it. 

```python
# Step1: Create a function where you can do operations on the model.

@exp.log_experiment   # This is pretty much it. This will log the experiment.
def attention_experiment():
    print("This is a placeholder for the experiment. Use as is.")
    return 4

# Step 2: Then, you run the function, get results. This starts the experiment.
attention_experiment()

# Step 3: Then, we will look at some of the things that logs keep a track of
l = exp.list_versions("attention_experiment")  # This gives the hash of the content of the experiment.
print("This is the version hash of the experiment: ", l)

# Step 4: You can also get the content of the experiment from the saved json.
print(exp.get_version("attention_experiment", l[0])['source'])  # This gives the content of the experiment.

```
Apart from these tools, there are also `@exp.profile_model`(to profile how much resources the model is using) and `@exp.test_hypothesis`(to test hypothesis). Support of more tools will be added as I get more feedback from the community.

### Step 4: Create you experiments

By default, Arrakis provides a lot of Anthropic's interpretability experiments(Monosemanticity, Residual Decomposition, Read Write Analysis and a lot [more]()). These are provided as tools, so in your experiments, you can plug and play with them and conduct your experiments. Here's an example of how you can do that.
```python
# Making functions for Arrakis to use is pretty easy. Let's look it in action.

# Step 1: Create a function where you can do operations on the model. Think of all the tools you might need for it.
# Step 2: Use the @exp.use_tools decorator on it, with additional arg of the tool.
# Step 3: The extra argument gives you access to the function. Done.

@exp.use_tools("write_read")  # use the `exp.use_tools()` decorator.
def read_write_analysis(read_layer_idx, write_layer_idx, src_idx, write_read=None):  # pass an additional argument.
    # Multi-hop attention (write-read)

    # use the extra argument as a tool.
    write_heads = write_read.identify_write_heads(read_layer_idx)  
    read_heads = write_read.identify_read_heads(write_layer_idx, dim_idx=src_idx) 

    return {
        "write_heads": write_heads, 
        "read_heads": read_heads
    }

print(read_write_analysis(0, 1, 0)) # Perfecto!

```

### Step 5: Visualize the Results
Generating plots is Arrakis is also plu and play, just add the decorator and plots are generated by default. Read more about the graphing docs [here]() 
```python

from arrakis.src.graph.base_graph import *

# Step 1: Create a function where you can want to draw plot.
# Step2: Use the @exp.plot_results decorator on it(set the plotting lib), with additional arg of the plot spec. Pass input_ids here as well(have to think on this)
# Step3: The extra argument gives you access to the fig. Done.

exp.set_plotting_lib(MatplotlibWrapper) # Set the plotting library.

@exp.plot_results(PlotSpec(plot_type = "attention", data_keys = "h.1.attn.c_attn"), input_ids=input_ids) # use the `exp.plot_results()` decorator.
def attention_heatmap(fig=None): # pass an additional argument.
    return fig

attention_heatmap() # Done.
plt.show()

```
These are three upper level classes in Arrakis. One is the `InterpretabilityBench` where you conduct experiments, the second is the `core_arrakis` where I've implemented some common tests for Transformer based model and the third is the `Graphing`.

## List of Tools
There is a lot of what's happening inside the `core_arrakis`. There are a lot of tools that we can use, which we'll deal with one by one. We'll understand what they do and how to use Arrakis to test them. These tools are supported as of now(please contribute more!)

- [Attention Head Composition](docs/AttentionHeadComposition.md)
- [Attention Tools](docs/AttentionTools.md)
- [Causal Tracing Intervention](docs/CausalTracingIntervention.md)
- [Knowledge Graph Extractor](docs/KnowledgeGraphExtractor.md)
- [Knowledge Prober](docs/KnowledgeProber.md)
- [Logit Attribution](docs/LogitAttribution.md)
- [Logit Lens](docs/LogitLens.md)
- [Read Write Heads](docs/ReadWriteHeads.md)
- [Residual Decomposition](docs/ResidualDecomposer.md)
- [Residual Tools](docs/ResidualTools.md)
- [Sparsity Analyzer](docs/SparsityAnalyzer.md)
- [Superposition Disentangler](docs/SuperpositionDisentangler.md)

Go to their respective pages and read about what they mean and how to use Arrakis to conduct experiments.

## Extending Arrakis
Apart from all of these tool, it is easy to develop tools on your own which you can use for your experiment. These are the steps to do so:
- Step 1: Make a class which inherits from the BaseInterpretabilityTool
```python
from arrakis.src.core_arrakis.base_interpret import BaseInterpretabilityTool

class CustomTool(BaseInterpretabilityTool):
    def __init__(self, model):
        super().__init__(model)
        self.model = model

    def custom_function(self, *args, **kwargs):
        # do some computations
        pass

    def another_custom_function(self, *args, **kwargs):
        # do another calcualtions
        pass 
```
The attribute `model` is a wrapper around Huggingface `PreTrainedModel` with many additional features which makes easier for experimentation purposes. The reference for model is given [here](). Write your function that utilizes the ActivationCache and get the intermediate activations.

- Step 2: In the derived class from `BaseInterpretabilityBench`, add your custom tool in the following manner.

```python
from src.bench.base_bench import BaseInteroretabilityBench
# Import the custom tool here.

class ExperimentBench(BaseInterpretabilityBench):
    def __init__(self, model, save_dir="experiments"):
        super().__init__(model, save_dir)
        self.tools.update({"custom": CustomTool(model)})

exp = ExperimentBench(model)  # where model is an instance of HookedAutoModel
```
And that is pretty much it. Now, in order to use it in a function, just do the following:

```python

@exp.use_tools("custom")
def test_custom_function(args, kwargs, custom): # the final argument should be the same name as the tool key. 
    custom.custom_function()
    custom.another_custom_function()

test_custom_function(args, kwargs)
```
Adding your own tool is really easy in Arrakis. Read the API reference guide to see how to implement your own functions. Open a PR for tools that are not implemented and I can add it quickly.

## How to Start?
For just starting out, consider going through the files `demo.ipynb` to get an overview of the library, `test_graphs.py` and `test_new_model.py` to test the model and the graphs(run from the command line)