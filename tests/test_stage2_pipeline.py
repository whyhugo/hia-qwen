import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hia_qwen.data import HIA_PHONE_TOKEN, HIA_UTT_TOKEN, HIA_WORD_TOKEN
from hia_qwen.joint_modeling import HiaQwenJointModel, is_stage2_trainable_name
from hia_qwen.schema import parse_json_prediction, render_json_target
from hia_qwen.stage2_data import Stage2JsonDataset

from test_stage1_alignment import FakeTokenizer, TinyLM


class TinyCacheLM(torch.nn.Module):
    def __init__(self, vocab_size=32, hidden_size=8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.head = torch.nn.Linear(hidden_size, vocab_size)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[7] = 10.0

    def get_input_embeddings(self):
        return self.embed

    def generate(self, *args, **kwargs):
        raise AssertionError("generate() must not be called for inputs_embeds decoding")

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, past_key_values=None, use_cache=False, return_dict=True, labels=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        logits = self.head(inputs_embeds)
        return type("Out", (), {"logits": logits, "past_key_values": past_key_values})()


class DecodeTokenizer(FakeTokenizer):
    eos_token_id = None

    def batch_decode(self, rows, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        inv = {v: k for k, v in self.vocab.items()}
        return [" ".join(inv.get(int(tok), "<unk>") for tok in row) for row in rows]


class Stage2PipelineTests(unittest.TestCase):
    def _first_multi_record(self):
        path = Path("/datas/store163/whyhugo/ms-qwen2-reproduce/data/processed/so762/train_multi_all.jsonl")
        if not path.exists():
            self.skipTest("multi_all JSONL is not available")
        return json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    def test_json_target_renderer_and_parser_roundtrip(self):
        record = self._first_multi_record()
        target = render_json_target(record)
        parsed = parse_json_prediction(target, record["labels"]["words"])
        self.assertEqual(set(parsed["sentence"].keys()), {"accuracy", "fluency", "prosody", "completeness", "total"})
        self.assertEqual(len(parsed["words"]), len(record["labels"]["words"]))
        self.assertEqual(
            len(parsed["phones"]),
            sum(len(word["phones"]) for word in record["labels"]["words"]),
        )

    def test_json_parser_rejects_bad_word_count(self):
        record = self._first_multi_record()
        bad = json.loads(render_json_target(record))
        bad["words"] = bad["words"][:-1]
        with self.assertRaises(ValueError):
            parse_json_prediction(json.dumps(bad), record["labels"]["words"])

    def test_stage2_dataset_aligns_multi_all_rows(self):
        jsonl = Path("/datas/store163/whyhugo/ms-qwen2-reproduce/data/processed/so762/train_multi_all.jsonl")
        seq_data = Path("/datas/store163/whyhugo/apa-hia-framework/data/seq_data_librispeech")
        raw_root = Path("/datas/store163/whyhugo/datasets/speechocean762")
        if not jsonl.exists() or not seq_data.exists() or not raw_root.exists():
            self.skipTest("SO762 processed data is not available")
        dataset = Stage2JsonDataset(jsonl, seq_data, raw_root, max_records=2)
        sample = dataset[0]
        self.assertEqual(sample["id"], "000010011")
        self.assertIn('"sentence"', sample["target"])
        self.assertEqual(tuple(sample["gop"].shape), (50, 84))

    def test_stage2_trainable_name_policy(self):
        self.assertTrue(is_stage2_trainable_name("projectors.word_projector.0.weight"))
        self.assertTrue(is_stage2_trainable_name("llm.model.layers.0.self_attn.q_proj.lora_A.default.weight"))
        self.assertFalse(is_stage2_trainable_name("llm.model.layers.0.self_attn.q_proj.base_layer.weight"))

    def test_projector_checkpoint_save_and_load(self):
        tokenizer = FakeTokenizer()
        model = HiaQwenJointModel(TinyLM(vocab_size=32, hidden_size=8), tokenizer, hia_dim=4, projector_hidden_dim=8)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_projectors(tmp)
            other = HiaQwenJointModel(TinyLM(vocab_size=32, hidden_size=8), tokenizer, hia_dim=4, projector_hidden_dim=8)
            other.load_projectors(Path(tmp) / "projector.pt")
            for key, tensor in model.projectors.state_dict().items():
                self.assertTrue(torch.equal(tensor, other.projectors.state_dict()[key]))

    def test_generate_json_uses_manual_inputs_embeds_decode(self):
        tokenizer = DecodeTokenizer()
        model = HiaQwenJointModel(TinyCacheLM(vocab_size=32, hidden_size=8), tokenizer, hia_dim=4, projector_hidden_dim=8)
        from hia_qwen.hia_features import HiaFeatureOutput
        features = HiaFeatureOutput(
            phone_features=[torch.randn(1, 4)],
            word_features=[torch.randn(1, 4)],
            utt_features=[torch.randn(1, 4)],
            raw_word_branch=torch.randn(1, 1, 4),
            phone_lengths=[1],
            word_lengths=[1],
        )
        prompt = f"a {HIA_UTT_TOKEN} b {HIA_WORD_TOKEN} c {HIA_PHONE_TOKEN}"
        decoded = model.generate_json([prompt], features, max_new_tokens=3)
        self.assertEqual(decoded, ["a a a"])


if __name__ == "__main__":
    unittest.main()
