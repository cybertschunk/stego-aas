from django.test import TestCase
from transformers import AutoTokenizer, GPT2LMHeadModel

from ..token_graph import build_token_graph, all_tokenizations, best_next_tokenization


class SparSampTest(TestCase):

    def test_graph_build(self):
        sample_text = "This is a short text for testing tokenization."
        graph = build_token_graph(sample_text)

        print(f"Graph built for {len(sample_text)} characters.")
        print(f"Total edges: {sum(len(v) for v in graph.values())}")

        # Show up to 10 tokenization paths
        for k, seq in enumerate(all_tokenizations(graph, len(sample_text))):
            print(f"Path {k + 1}: {seq}")

    def test_token_prediction(self):
        text = "Autonomous systems are cool."
        tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=False)
        model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

        graph = build_token_graph(text, tokenizer)  # from earlier
        pref = tokenizer.encode("Autonomous")  # example *correct* prefix
        neg1 = tokenizer.encode(" systems")  # forbid this immediate next
        neg2 = tokenizer.encode(" systems are")  # and this longer one

        best_next = best_next_tokenization(
            text,
            graph,
            pref,
            [neg1, neg2],
            model,
            tokenizer,
        )

        print("Chosen token ids:", best_next)
        print("Decodes to:", tokenizer.decode(best_next))




