"""3 encoder dùng chung cho cả lúc build pkl lẫn lúc suy luận (để feature giống nhau tuyệt đối).

TextEncoder : RoBERTa-base -> <s> (768) -> Linear(768->1024)          [fine-tune bằng text_finetune.py]
AudioEncoder: Wav2Vec2-base (đóng băng) -> mean theo thời gian (768) -> Linear(768->300) [PCA gấp vào Linear]
FaceEncoder : DenseNet-121 FER+ -> embedding 342 (checkpoint từ densenet_fer.py)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import SR, decode_audio
import os


# ============================================================================ TEXT
class TextEncoder(nn.Module):
    def __init__(self, base="roberta-base", out_dim=1024, n_cls=7, pretrained=True):
        super().__init__()
        from transformers import RobertaConfig, RobertaModel

        if pretrained:
            self.bert = RobertaModel.from_pretrained(base, add_pooling_layer=False)
        else:  # chỉ lấy cấu trúc, trọng số sẽ nạp từ checkpoint
            self.bert = RobertaModel(RobertaConfig.from_pretrained(base), add_pooling_layer=False)
        self.proj = nn.Linear(self.bert.config.hidden_size, out_dim)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(out_dim, n_cls)
        self.base, self.out_dim = base, out_dim

    def features(self, input_ids, attention_mask):
        h = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.proj(h)  # [B, out_dim]  <- đưa vào DiGEMO

    def forward(self, input_ids, attention_mask):
        f = self.features(input_ids, attention_mask)
        return self.fc(self.drop(F.gelu(f))), f

    @classmethod
    def load(cls, ckpt, device="cpu"):
        ck = torch.load(ckpt, map_location="cpu")
        state = ck["model"]
        # MELD checkpoints may omit n_cls; infer it from the classifier.
        n_cls = int(ck.get("n_cls", state["fc.weight"].shape[0]))
        m = cls(ck["base"], ck["out_dim"], n_cls=n_cls, pretrained=False)
        m.load_state_dict(state)
        return m.to(device).eval()


@torch.no_grad()
def text_features(model, tok, texts, device, bs=64, max_len=128):
    model.eval()
    out = np.zeros((len(texts), model.out_dim), np.float32)
    order = np.argsort([len(t) for t in texts])  # gom câu cùng độ dài -> ít padding
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        out[idx] = model.features(enc["input_ids"], enc["attention_mask"]).float().cpu().numpy()
    return out


# ============================================================================ AUDIO
def load_wave(path):
    wav, rc = decode_audio(path)
    if rc != 0 or wav.size == 0:
        print(f"[WARN] không đọc được audio: {path} (dùng im lặng 0.1s)")
    wav = torch.from_numpy(wav)
    if wav.numel() < SR // 10:
        wav = F.pad(wav, (0, SR // 10 - wav.numel()))
    return (wav - wav.mean()) / torch.sqrt(wav.var() + 1e-7)


class _WavDS(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_wave(self.paths[i])


def _collate(batch):
    lens = torch.tensor([len(w) for w in batch])
    x = torch.zeros(len(batch), int(lens.max()))
    for i, w in enumerate(batch):
        x[i, : len(w)] = w
    return x, lens


def fold_pca(X, n=300):
    """StandardScaler + PCA(n) gấp thành (W [n,D], b [n]) sao cho y = X @ W.T + b."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(X)
    pca = PCA(n_components=n, random_state=0).fit(sc.transform(X))
    C = pca.components_
    W = C / sc.scale_[None, :]
    b = -((sc.mean_ / sc.scale_) + pca.mean_) @ C.T
    return W.astype(np.float32), b.astype(np.float32)


class AudioEncoder(nn.Module):
    def __init__(self, w2v="facebook/wav2vec2-base"):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.w2v = Wav2Vec2Model.from_pretrained(w2v).eval()
        self.W = self.b = None

    @torch.no_grad()
    def _pool(self, x, lens):
        h = self.w2v(x).last_hidden_state
        flen = self.w2v._get_feat_extract_output_lengths(lens).clamp(min=1)
        mask = torch.arange(h.size(1), device=h.device)[None, :] < flen[:, None]
        return (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)

    @torch.no_grad()
    def raw768(self, paths, bs=8, workers=0):
        """list đường dẫn -> np [N,768] (giữ đúng thứ tự đầu vào)."""
        dev = next(self.w2v.parameters()).device
        order = sorted(range(len(paths)), key=lambda i: os.path.getsize(paths[i]))
        loader = DataLoader(_WavDS([paths[i] for i in order]), bs, shuffle=False,
                            num_workers=workers, collate_fn=_collate)
        out = np.zeros((len(paths), 768), np.float32)
        pos = 0
        for x, lens in loader:
            e = self._pool(x.to(dev), lens.to(dev)).float().cpu().numpy()
            out[order[pos:pos + len(e)]] = e
            pos += len(e)
        return out

    def set_head(self, W, b):
        self.W, self.b = np.asarray(W, np.float32), np.asarray(b, np.float32)

    def load_head(self, path):
        d = np.load(path)
        self.set_head(d["W"], d["b"])

    def project(self, raw):
        return (raw @ self.W.T + self.b).astype(np.float32)


# ============================================================================ OPENSMILE AUDIO
class OpenSmileAudioEncoder:
    """openSMILE IS10 functionals -> 1582-dim normalized audio features.

    This class is separate from AudioEncoder so MELD keeps its existing
    Wav2Vec2 + PCA(300) behavior unchanged.
    """
    dim = 1582

    def __init__(self, a_mean=None, a_std=None):
        import opensmile

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.IS10,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        actual_dim = len(self.smile.feature_names)
        if actual_dim != self.dim:
            raise RuntimeError(f"Expected {self.dim} openSMILE features, got {actual_dim}")
        self.a_mean = None if a_mean is None else np.asarray(a_mean, np.float32)
        self.a_std = None if a_std is None else np.maximum(np.asarray(a_std, np.float32), 1e-6)

    def extract_signal(self, wav, sr=16000):
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        wav = wav[: int(sr * 15)]
        if wav.size < sr // 10:
            wav = np.pad(wav, (0, sr // 10 - wav.size))
        feat = self.smile.process_signal(wav, sr).to_numpy().reshape(-1)
        return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def extract_file(self, path):
        wav, rc = decode_audio(path)
        if rc != 0 or wav.size == 0:
            wav = np.zeros(SR // 10, dtype=np.float32)
        return self.extract_signal(wav, SR)

    def set_norm(self, a_mean, a_std):
        self.a_mean = np.asarray(a_mean, np.float32)
        self.a_std = np.maximum(np.asarray(a_std, np.float32), 1e-6)

    def normalize(self, raw):
        if self.a_mean is None or self.a_std is None:
            raise RuntimeError("Audio normalization statistics are not set")
        raw = np.asarray(raw, np.float32)
        return ((raw - self.a_mean) / self.a_std).astype(np.float32)


# ============================================================================ FACE
class _FrameDS(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


def pick_frames(paths, n):
    """chọn n frame cách đều (n<=0 hoặc ít hơn n frame -> dùng hết)."""
    if n <= 0 or len(paths) <= n:
        return list(paths)
    idx = np.unique(np.linspace(0, len(paths) - 1, n).round().astype(int))
    return [paths[i] for i in idx]


class FaceEncoder:
    def __init__(self, ckpt, device="cpu"):
        from densenet_fer import DenseNetFER, eval_tf

        ck = torch.load(ckpt, map_location="cpu")
        self.model = DenseNetFER(len(ck["classes"]), emb_dim=ck["emb_dim"], pretrained=False)
        self.model.load_state_dict(ck["model"])
        self.model.to(device).eval()
        self.tf, self.device, self.dim = eval_tf(), device, ck["emb_dim"]

    @torch.no_grad()
    def _run(self, loader):
        out = []
        for x in loader:
            out.append(self.model.embed(x.to(self.device)).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def embed_paths(self, paths, bs=128, workers=4):
        return self._run(DataLoader(_FrameDS(paths, self.tf), bs, shuffle=False, num_workers=workers))

    def embed_pil(self, imgs, bs=64):
        """list PIL RGB -> [N,342]"""
        class DS(Dataset):
            def __len__(s):
                return len(imgs)

            def __getitem__(s, i):
                return self.tf(imgs[i].convert("RGB"))

        return self._run(DataLoader(DS(), bs, shuffle=False))
