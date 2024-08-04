# Add models from wherever you want, and run from cli to test where the prod is breaking.

from arrakis.src.bench.base_bench import BaseInterpretabilityBench
from arrakis.src.graph.base_graph import *
from arrakis.src.core_arrakis.activation_cache import *
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(prog="Test new Model for all the tools.")
    parser.add_argument("--model_name", type=str, default="llama")
    parser.add_argument("--vocab_size", type=int, default=50256)
    parser.add_argument("--hidden_size", type=int, default=8)  # embed_dim
    parser.add_argument("--intermediate_size", type=int, default=4)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)

    TOOL_CHOICES = ["causal", "knowledge", "logit", "attention", "write_read", "composer", "kg_extractor", "logit_lens", "decomposer", "sparsity", "superposition", "residual", "k_prober", "surgery", "feature_viz"]
    parser.add_argument("--tool", type=str, 
                        default=TOOL_CHOICES, 
                        choices=TOOL_CHOICES, 
                        nargs="+")


    args = parser.parse_args()
    
    config = HookedAutoConfig(name=args.model_name, 
            vocab_size=args.vocab_size, 
            hidden_size=args.hidden_size, 
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.num_hidden_layers, 
            num_attention_heads=args.num_attention_heads, 
            num_key_value_heads=args.num_key_value_heads)
    
    model = HookedAutoModel(config)

    input_ids = torch.randint(0, 50256, (1, 50))


    class AttentionVisulazierExperiment(BaseInterpretabilityBench):
        def __init__(self, model, save_dir="experiments"):
            super().__init__(model, save_dir)

    exp = AttentionVisulazierExperiment(model)


    # trace_token_probability(self, input_ids, target_idx)
    # intervene_on_neuron(self, input_ids, layer_idx, neuron_idx, scale=0)
    @exp.use_tools("causal")
    def test_causal_attention(input_ids, layer_idx, neu_idx, scale, causal):
        ttp = causal.trace_token_probability(input_ids, 0)
        intervene = causal.intervene_on_neuron(input_ids, layer_idx, neu_idx, scale)
        return {
            "intervene": intervene,
            "trace_token_probability": ttp
        }


    @exp.use_tools("knowledge")
    def test_knowledge_prober(input_ids, layer_idx, neuron_idx, knowledge):
        pscore = knowledge.polysemantic_score(layer_idx, 2)
        true_false = knowledge.knowledge_probe([(input_ids, input_ids)], (layer_idx, neuron_idx))
        return {
            "k_probe": true_false,
            "polysemantic_score": pscore

        }

    # logit_attribution(self, input_ids, target_idx)
    # track_token_circulation(self, input_ids, target_idx, n_layers=2)
    @exp.use_tools("logit")
    def test_logit_attributor(input_ids, target_idx, n_layers, logit):
        direct_moves = logit.track_token_circulation(input_ids, target_idx, n_layers)
        logit_attribution = logit.logit_attribution(input_ids, target_idx)
        return {
            "direct_moves": direct_moves,
            "logit_attribution": logit_attribution
        }

    # attention_patterns(self, input_ids, layer_idx, head_idx):
    # top_attended_ids(self, input_ids, layer_idx, head_idx, num_tokens=5)
    # ablate_head(self,layer_idx, head_idx):
    # merge_heads(self, layer_idx, head_idx1, head_idx2, alpha=0.5):
    @exp.use_tools("attention")
    def test_attention_tools(layer_idx, head_idx1, head_idx2, alpha, attention):
        ablate_head = attention.ablate_head(layer_idx, head_idx1)
        merge_heads = attention.merge_heads(layer_idx, head_idx1, head_idx2, alpha)
        attention_patterns = attention.attention_patterns(input_ids, layer_idx, head_idx1)
        top_attended_ids = attention.top_attended_ids(input_ids, layer_idx, head_idx1, 1)
        return {
            "ablate_head": ablate_head,
            "merge_heads": merge_heads,
            "attention_patterns": attention_patterns,
            "top_attended_ids": top_attended_ids
        }


    @exp.use_tools("write_read") # Works with the new setup as well.
    def test_read_write(src_idx, write_read):
        write_heads = write_read.identify_write_heads(0)  # Example layer
        read_heads = write_read.identify_read_heads(1, dim_idx=src_idx)  # Example layer
        return {
            "write_heads": write_heads,
            "read_heads": read_heads
        }

    @exp.use_tools("composer")
    def test_attention_composer(head1, head2, alpha, composer):
        hc = composer.head_composition(head1, head2, alpha)
        app = composer.attention_path_patching(input_ids, head1, head2)
        return {
            "head_composition": hc,
            "attention_path_patching": app
        }

    # Retiring  as of now, getting embeddings for model will require me to change ModelAttributes a litlle bit. 
    @exp.use_tools("kg_extractor")
    def test_kg_extractor(token_ids, kg_extractor):
        top_subspaces = kg_extractor.extract_relational_subspaces(0)
        irt = kg_extractor.identify_relation_types(token_ids, top_subspaces)
        return {
            "top_subspaces": top_subspaces,
            "identify_relation_types": irt
        }


    @exp.use_tools("logit_lens")
    def test_logit_lens(input_ids, target_idx, layer_idx, logit_lens):
        ll = logit_lens.logit_lens(input_ids, target_idx, layer_idx)
        la = logit_lens.layer_attributions(input_ids, target_idx)
        return {
            "logit_lens": ll,
            "layer_attributions": la
        }

    @exp.use_tools("decomposer")
    def test_residual_decomposer(input_ids, layer_idx, decomposer):
        rd = decomposer.decompose_residual(input_ids, layer_idx)
        return rd
    
    @exp.use_tools("sparsity")
    def test_sparsity_analyzer(input_ids, layer="layers.1.self_attn.hook_result_post",  sparsity=None):
        act_sparsity = sparsity.compute_activation_sparsity([input_ids], layer)
        find_ps_neu = sparsity.find_polysemantic_neurons(3)
        prune_network = sparsity.prune_network(0.9)
        return {
            "act_sparsity": act_sparsity,
            "find_ps_neu": find_ps_neu,
            "prune_network": prune_network
        }

    @exp.use_tools("superposition")
    def test_superposition_disentangler(input_ids, layer_idx, superposition):
        act_covariance = superposition.compute_activation_covariance([input_ids], layer_idx)
        dis_supoerpos = superposition.disentangle_superposition(act_covariance)
        return {
            "act_covariance": act_covariance,
            "dis_supoerpos": dis_supoerpos
        }
    
    @exp.use_tools("residual")
    def test_residual_tools(layer_idx, residual):
        rd = residual.residual_decomposition(input_ids, layer_idx)
        rm = residual.residual_movement(input_ids, 0, 1)
        fa = residual.feature_attribution(input_ids, 0)
        return {
            "residual_decomposition": rd,
            "residual_movement": rm,
            "feature_attribution": fa
        }

    @exp.use_tools("k_prober")
    def test_knowledge_prober(ids, layer_idx, neuron_idx, k_prober):
        pscore = k_prober.polysemantic_score(layer_idx, 2)
        true_false = k_prober.knowledge_probe([(ids, ids)], (layer_idx, neuron_idx))
        return {
            "k_probe": true_false,
            "polysemantic_score": pscore
        }
    
    @exp.use_tools("surgery")
    def test_model_surgery(indicies, replacements, surgery):
        model = surgery.get_model()
        print("Before surgery: \n\n",model)

        with surgery.replace_layers(indicies, replacements):
            print("After surgery: \n\n",model)
        # print(surgery.permute_layers([0, 1]))
        with surgery.permute_layers([0, 1]):
            print("After surgery: \n\n",model)

        with surgery.delete_layers([0, 1]):
            print("After surgery: \n\n",model)
    
    @exp.use_tools("feature_viz")
    def test_feature_viz(feature_viz):
        top_k = feature_viz.activate_neuron([input_ids],"layers.1.self_attn.q_proj.hook_result_pre", 0, top_k=10)
        interpolations = feature_viz.neuron_interpolation(input_ids, "model.layers.1", 0, 1, num_interpolations=10, max_new_tokens=10)
        return {"top_k": top_k, "interpolations": interpolations}

    # based on whatever tools we want to test, we can call the functions.
    _, acts = model(input_ids)
    # print(acts.keys())
    # print(model.model.get_output_embeddings)

    if 'causal' in args.tool:
        print(test_causal_attention(input_ids, 0, 0, 0.5))
    if 'knowledge' in args.tool:
        print(test_knowledge_prober(input_ids, 0, 0))
    if 'logit' in args.tool:
        print(test_logit_attributor(input_ids, 0, 2))
    if 'attention' in args.tool:
        print(test_attention_tools(0, 0, 1, 0.5))
    if 'write_read' in args.tool:
        print(test_read_write(0))
    if 'composer' in args.tool:
        print(test_attention_composer((0, 0), (0, 1), 0.5))
    if 'kg_extractor' in args.tool:
        print(test_kg_extractor(input_ids))
    if 'logit_lens' in args.tool:
        print(test_logit_lens(input_ids, 0, 0))
    if 'decomposer' in args.tool:
        print(test_residual_decomposer(input_ids, 0))
    if 'sparsity' in args.tool:
        print(test_sparsity_analyzer(input_ids, layer="layers.0.self_attn.hook_result_post"))
        print(test_sparsity_analyzer(input_ids, layer="layers.1.self_attn.hook_result_post"))
        print(test_sparsity_analyzer(input_ids, layer="layers.2.self_attn.hook_result_post"))
    if 'superposition' in args.tool:
        print(test_superposition_disentangler(input_ids, 'layers.0.self_attn.hook_result_post'))
    if 'residual' in args.tool:
        print(test_residual_tools(0))
    if 'k_prober' in args.tool:
        print(test_knowledge_prober(input_ids, 0, 0))
    if 'surgery' in args.tool:
        print(test_model_surgery([0, 1], [model.model_attrs.get_mlp_out(0), model.model_attrs.get_mlp_out(1)]))
    if 'feature_viz' in args.tool:
        print(test_feature_viz())
        
    print("Completed. All tests ran successfully.")