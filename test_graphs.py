from src.bench.base_bench import BaseInterpretabilityBench
from src.graph.base_graph import *
from pre_trained_models_hf import *
from argparse import ArgumentParser

if __name__ == "__main__":
    arg_parser = ArgumentParser(prog="Test new Model for all the tools.")
    arg_parser.add_argument("--model_name", type=str, default="llama")
    arg_parser.add_argument("--vocab_size", type=int, default=50256)
    arg_parser.add_argument("--hidden_size", type=int, default=4)
    arg_parser.add_argument("--intermediate_size", type=int, default=2)
    arg_parser.add_argument("--num_hidden_layers", type=int, default=4)
    arg_parser.add_argument("--num_attention_heads", type=int, default=4)
    arg_parser.add_argument("--num_key_value_heads", type=int, default=4)

    GRAPH_CHOICES = ["attention", "neuron_activation", "residual_stream", "layer_norms", "induction_heads", "neuron_activation_trajectories", "qk_dot_product"]
    arg_parser.add_argument("--graph", type=str, default=GRAPH_CHOICES, choices=GRAPH_CHOICES, nargs="+")
    args = arg_parser.parse_args()

    config = HookedAutoConfig(name=args.model_name, vocab_size=args.vocab_size, hidden_size=args.hidden_size, intermediate_size=args.intermediate_size, num_hidden_layers=args.num_hidden_layers, num_attention_heads=args.num_attention_heads, num_key_value_heads=args.num_key_value_heads)
    model = HookedAutoModel(config)

    input_ids = torch.randint(0, 50256, (1, 10))

    class AttentionVisulazierExperiment(BaseInterpretabilityBench):
        def __init__(self, model, save_dir="experiments"):
            super().__init__(model, save_dir)

    exp = AttentionVisulazierExperiment(model)
    exp.set_plotting_lib(MatplotlibWrapper)

    @exp.plot_results(PlotSpec(plot_type = "attention", data_keys = ["layers.1.self_attn.q_proj.hook_result_pre"]), input_ids=input_ids)
    def visualize_attention_graphs(fig=None):
        return fig

    @exp.plot_results(PlotSpec(plot_type = "neuron_activation", data_keys = ["layers.1.mlp.gate_proj"]), input_ids=input_ids)
    def visualize_neuron_activations(fig=None):
        return fig

    @exp.plot_results(PlotSpec(plot_type = "residual_stream", data_keys = None), input_ids=input_ids)
    def visualize_residual_stream(fig=None):
        return fig


    @exp.plot_results(PlotSpec(plot_type = "layer_norms", data_keys = None), input_ids=input_ids)
    def visualize_layer_norms(fig=None):
        return fig

    @exp.plot_results(PlotSpec(plot_type = "induction_heads", data_keys = ['layers.0.self_attn.q_proj.hook_result_post']), input_ids=input_ids)
    def visualize_induction_heads(fig=None):
        return fig


    @exp.plot_results(PlotSpec(plot_type = "neuron_activation_trajectories", data_keys = ["layers.3.mlp.hook_result_post"]), input_ids=input_ids)
    def visualize_neuron_activation_trajectories(fig=None):
        return fig

    @exp.plot_results(PlotSpec(plot_type = "qk_dot_product", data_keys = ["layers.1.self_attn.q_proj.hook_result_post", "layers.1.self_attn.k_proj.hook_result_post"]), input_ids=input_ids)
    def visualize_qk_dot_product(fig=None):
        return fig

    if "attention" in args.graph:
        visualize_attention_graphs()
    
    if "neuron_activation" in args.graph:
        visualize_neuron_activations()
    
    if "residual_stream" in args.graph:
        visualize_residual_stream()
    
    if "layer_norms" in args.graph:
        visualize_layer_norms()
    
    if "induction_heads" in args.graph:
        visualize_induction_heads()
    
    if "neuron_activation_trajectories" in args.graph:
        visualize_neuron_activation_trajectories()
    
    if "qk_dot_product" in args.graph:
        visualize_qk_dot_product()
    
    plt.show()