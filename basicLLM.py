import argparse
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets

manualDataset = False
randomSeed = True

if manualDataset:
    with open("trainingText.txt", "r", encoding="utf-8") as f:
        DEFAULT_TRAINING_TEXT = f.read()
else:
	dataset = datasets.load_dataset("openbmb/UltraInteract_sft", split="train")
	DEFAULT_TRAINING_TEXT = dataset["instruction"][0] + "\n\n" + dataset["response"][500]

def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)


def build_vocab(text: str):
	chars = sorted(list(set(text)))
	stoi = {ch: i for i, ch in enumerate(chars)}
	itos = {i: ch for ch, i in stoi.items()}
	return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
	return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode(token_ids: torch.Tensor, itos: dict[int, str]) -> str:
	return "".join(itos[i] for i in token_ids.tolist())


@dataclass
class Config:
	batch_size: int = 32
	block_size: int = 64
	max_iters: int = 1500
	eval_interval: int = 200
	eval_iters: int = 50
	learning_rate: float = 3e-4
	n_embd: int = 96
	n_head: int = 4
	n_layer: int = 3
	dropout: float = 0.15
	seed: int = 42
	generate_tokens: int = 250


class Head(nn.Module):
	def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias=False)
		self.query = nn.Linear(n_embd, head_size, bias=False)
		self.value = nn.Linear(n_embd, head_size, bias=False)
		self.dropout = nn.Dropout(dropout)
		self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		_, time, _ = x.shape
		k = self.key(x)
		q = self.query(x)
		head_size = k.size(-1)
		wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)
		wei = wei.masked_fill(self.tril[:time, :time] == 0, float("-inf"))
		wei = F.softmax(wei, dim=-1)
		wei = self.dropout(wei)
		v = self.value(x)
		out = wei @ v
		return out


class MultiHeadAttention(nn.Module):
	def __init__(self, n_head: int, n_embd: int, block_size: int, dropout: float):
		super().__init__()
		head_size = n_embd // n_head
		self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(n_head)])
		self.proj = nn.Linear(n_embd, n_embd)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = torch.cat([head(x) for head in self.heads], dim=-1)
		out = self.proj(out)
		out = self.dropout(out)
		return out


class FeedForward(nn.Module):
	def __init__(self, n_embd: int, dropout: float):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, 4 * n_embd),
			nn.GELU(),
			nn.Linear(4 * n_embd, n_embd),
			nn.Dropout(dropout),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class Block(nn.Module):
	def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
		super().__init__()
		self.sa = MultiHeadAttention(n_head, n_embd, block_size, dropout)
		self.ffwd = FeedForward(n_embd, dropout)
		self.ln1 = nn.LayerNorm(n_embd)
		self.ln2 = nn.LayerNorm(n_embd)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.sa(self.ln1(x))
		x = x + self.ffwd(self.ln2(x))
		return x


class TinyGPT(nn.Module):
	def __init__(self, vocab_size: int, cfg: Config):
		super().__init__()
		self.cfg = cfg
		self.token_embedding_table = nn.Embedding(vocab_size, cfg.n_embd)
		self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
		self.blocks = nn.Sequential(
			*[Block(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout) for _ in range(cfg.n_layer)]
		)
		self.ln_f = nn.LayerNorm(cfg.n_embd)
		self.lm_head = nn.Linear(cfg.n_embd, vocab_size)

	def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
		bsz, time = idx.shape
		tok_emb = self.token_embedding_table(idx)
		pos_emb = self.position_embedding_table(torch.arange(time, device=idx.device))
		x = tok_emb + pos_emb
		x = self.blocks(x)
		x = self.ln_f(x)
		logits = self.lm_head(x)

		loss = None
		if targets is not None:
			bsz, time, channels = logits.shape
			logits = logits.view(bsz * time, channels)
			targets = targets.view(bsz * time)
			loss = F.cross_entropy(logits, targets)

		return logits, loss

	@torch.no_grad()
	def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
		self.eval()
		for _ in range(max_new_tokens):
			idx_cond = idx[:, -self.cfg.block_size :]
			logits, _ = self(idx_cond)
			logits = logits[:, -1, :]
			probs = F.softmax(logits, dim=-1)
			idx_next = torch.multinomial(probs, num_samples=1)
			idx = torch.cat((idx, idx_next), dim=1)
		return idx


def get_batch(data: torch.Tensor, cfg: Config, device: str):
	if len(data) <= cfg.block_size + 1:
		raise ValueError(
			f"Dataset split too short for block_size={cfg.block_size}. "
			f"Need at least {cfg.block_size + 2} tokens, got {len(data)}."
		)
	ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
	x = torch.stack([data[i : i + cfg.block_size] for i in ix])
	y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])
	return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model: TinyGPT, train_data: torch.Tensor, val_data: torch.Tensor, cfg: Config, device: str):
	model.eval()
	out = {}
	for split, split_data in [("train", train_data), ("val", val_data)]:
		losses = torch.zeros(cfg.eval_iters)
		for k in range(cfg.eval_iters):
			xb, yb = get_batch(split_data, cfg, device)
			_, loss = model(xb, yb)
			losses[k] = loss.item()
		out[split] = losses.mean().item()
	model.train()
	return out


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train a tiny educational LLM (character-level Transformer)")
	parser.add_argument("--text-file", type=str, default="", help="Optional path to training text file")
	parser.add_argument("--max-iters", type=int, default=1500, help="Training iterations")
	parser.add_argument("--generate-tokens", type=int, default=250, help="Number of chars to generate")
	parser.add_argument("--seed", type=int, default=random.randint(), help="Random seed")
	parser.add_argument("--num-threads", type=int, default=7, help="PyTorch CPU threads to use")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	cfg = Config(max_iters=args.max_iters, generate_tokens=args.generate_tokens, seed=args.seed)
	if args.num_threads > 0:
		torch.set_num_threads(args.num_threads)
		if hasattr(torch, "set_num_interop_threads"):
			torch.set_num_interop_threads(max(1, min(2, args.num_threads)))
	set_seed(cfg.seed)

	if args.text_file:
		with open(args.text_file, "r", encoding="utf-8") as f:
			text = f.read()
	else:
		text = DEFAULT_TRAINING_TEXT

	if len(text) < 20:
		raise ValueError("Training text is too short. Provide at least 20 characters.")

	if len(text) < cfg.block_size + 2:
		cfg.block_size = len(text) - 2
		print(f"Adjusted block_size to {cfg.block_size} to match small dataset size.")

	stoi, itos = build_vocab(text)
	data = encode(text, stoi)
	split_idx = int(0.9 * len(data))
	train_data = data[:split_idx]
	val_data = data[split_idx:]
	if len(val_data) <= cfg.block_size + 1:
		val_data = train_data
		print("Validation split is too short for current block_size; using train split for validation estimates.")

	device = "cuda" if torch.cuda.is_available() else "cpu"
	model = TinyGPT(vocab_size=len(stoi), cfg=cfg).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

	total_params = sum(p.numel() for p in model.parameters())
	print(
		f"device={device}, vocab_size={len(stoi)}, parameters={total_params:,}, "
		f"threads={torch.get_num_threads()}"
	)

	for step in range(cfg.max_iters + 1):
		if step % cfg.eval_interval == 0:
			losses = estimate_loss(model, train_data, val_data, cfg, device)
			print(f"step {step:4d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

		xb, yb = get_batch(train_data, cfg, device)
		_, loss = model(xb, yb)
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()

	start_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
	generated = model.generate(start_idx, max_new_tokens=cfg.generate_tokens)[0]
	print("\n--- Generated Text ---")
	print(decode(generated, itos))


if __name__ == "__main__":
	main()
