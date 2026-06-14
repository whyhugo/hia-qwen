import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hia_qwen.data import AlignmentDataset, HIA_PHONE_TOKEN, HIA_UTT_TOKEN, HIA_WORD_TOKEN
from hia_qwen.hia_features import HiaFeatureExtractor, HiaFeatureOutput, pool_word_branch_by_word_id
from hia_qwen.modeling import HiaQwenAlignmentModel, merge_hia_soft_tokens


class FakeTokenizer:
    def __init__(self):
        tokens = [
            "<pad>",
            "<unk>",
            HIA_UTT_TOKEN,
            HIA_WORD_TOKEN,
            HIA_PHONE_TOKEN,
            "User:",
            "Assistant:",
            "a",
            "b",
            "c",
            "Score:",
            "8",
            "9",
        ]
        self.vocab = {tok: i for i, tok in enumerate(tokens)}
        self.pad_token_id = self.vocab["<pad>"]
        self.unk_token_id = self.vocab["<unk>"]
        self.eos_token = ""

    def __len__(self):
        return len(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def __call__(self, texts, return_tensors=None, padding=False):
        rows = []
        for text in texts:
            rows.append([self.convert_tokens_to_ids(tok) for tok in text.split()])
        max_len = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), max_len), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(rows), max_len), dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = torch.tensor(row)
            attention[i, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention}


class TinyLM(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        logits = self.head(inputs_embeds)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


class Stage1AlignmentTests(unittest.TestCase):
    def test_dataset_aligns_jsonl_with_hia_rows(self):
        jsonl = Path(
            "/datas/store163/whyhugo/ms-qwen2-reproduce/data/processed/so762/train_sentence_total.jsonl"
        )
        seq_data = Path("/datas/store163/whyhugo/apa-hia-framework/data/seq_data_librispeech")
        raw_root = Path("/datas/store163/whyhugo/datasets/speechocean762")
        if not jsonl.exists() or not seq_data.exists() or not raw_root.exists():
            self.skipTest("SO762 processed data is not available")
        dataset = AlignmentDataset(jsonl, seq_data, raw_root, max_records=2)
        sample = dataset[0]
        self.assertEqual(sample["id"], "000010011")
        self.assertEqual(tuple(sample["gop"].shape), (50, 84))
        self.assertEqual(tuple(sample["phn_id"].shape), (50,))
        self.assertIn(HIA_UTT_TOKEN, sample["prompt"])
        self.assertIn(HIA_WORD_TOKEN, sample["prompt"])
        self.assertIn(HIA_PHONE_TOKEN, sample["prompt"])

    def test_word_branch_pooling_preserves_dynamic_word_count(self):
        hidden = torch.arange(1 * 6 * 2, dtype=torch.float32).view(1, 6, 2)
        word_id = torch.tensor([[0, 0, 1, 2, 2, -1]])
        pooled = pool_word_branch_by_word_id(hidden, word_id)
        self.assertEqual(tuple(pooled[0].shape), (3, 2))
        self.assertTrue(torch.allclose(pooled[0][0], hidden[0, :2].mean(dim=0)))
        self.assertTrue(torch.allclose(pooled[0][1], hidden[0, 2]))
        self.assertTrue(torch.allclose(pooled[0][2], hidden[0, 3:5].mean(dim=0)))

    def test_hia_extractor_exposes_raw_word_branch_sequence(self):
        checkpoint = Path("/datas/store163/whyhugo/apa-hia-framework/exp/hia_seed17/models/best_audio_model.pth")
        if not checkpoint.exists():
            self.skipTest("HIA checkpoint is not available")
        extractor = HiaFeatureExtractor(
            hia_repo="/datas/store163/whyhugo/apa-hia-framework",
            checkpoint_path=checkpoint,
        )
        gop = torch.zeros(2, 50, 84)
        phn = torch.full((2, 50), -1, dtype=torch.long)
        phn[0, :5] = torch.tensor([1, 2, 3, 4, 5])
        phn[1, :3] = torch.tensor([1, 2, 3])
        word_id = torch.full((2, 50), -1, dtype=torch.long)
        word_id[0, :2] = 0
        word_id[0, 2:5] = 1
        word_id[1, :3] = 0
        features = extractor(gop, phn, word_id)
        self.assertEqual(tuple(features.raw_word_branch.shape), (2, 50, 48))
        self.assertEqual(features.word_lengths, [2, 1])
        self.assertEqual(tuple(features.word_features[0].shape), (2, 48))

    def test_soft_token_merge_expands_embeddings_and_labels(self):
        input_ids = torch.tensor([[7, 2, 8, 3, 9, 4, 10, 11]])
        attention = torch.ones_like(input_ids)
        labels = torch.tensor([[-100, -100, -100, -100, -100, -100, 10, 11]])
        embeds = torch.randn(1, input_ids.shape[1], 8)
        projected = {
            "utt": [torch.randn(1, 8)],
            "word": [torch.randn(2, 8)],
            "phone": [torch.randn(3, 8)],
        }
        merged = merge_hia_soft_tokens(
            input_ids,
            attention,
            labels,
            embeds,
            {HIA_UTT_TOKEN: 2, HIA_WORD_TOKEN: 3, HIA_PHONE_TOKEN: 4},
            projected,
        )
        self.assertEqual(tuple(merged["inputs_embeds"].shape), (1, 11, 8))
        self.assertEqual(tuple(merged["attention_mask"].shape), (1, 11))
        self.assertEqual(tuple(merged["labels"].shape), (1, 11))
        self.assertEqual(int((merged["labels"] != -100).sum().item()), 2)

    def test_mock_backward_only_projectors_trainable(self):
        tokenizer = FakeTokenizer()
        llm = TinyLM(vocab_size=32, hidden_size=8)
        model = HiaQwenAlignmentModel(llm=llm, tokenizer=tokenizer, hia_dim=4, projector_hidden_dim=8)
        prompts = [f"a {HIA_UTT_TOKEN} b {HIA_WORD_TOKEN} c {HIA_PHONE_TOKEN}"]
        targets = ["Score: 8"]
        features = HiaFeatureOutput(
            phone_features=[torch.randn(3, 4)],
            word_features=[torch.randn(2, 4)],
            utt_features=[torch.randn(1, 4)],
            raw_word_branch=torch.randn(1, 5, 4),
            phone_lengths=[3],
            word_lengths=[2],
        )
        loss = model(prompts, targets, features).loss
        loss.backward()
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        grads = [name for name, p in model.named_parameters() if p.grad is not None]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("projectors.") for name in trainable))
        self.assertTrue(grads)
        self.assertTrue(all(name.startswith("projectors.") for name in grads))


if __name__ == "__main__":
    unittest.main()
