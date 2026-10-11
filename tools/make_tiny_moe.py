#!/usr/bin/env python3
import argparse
import random
import struct

GGUF_UINT32, GGUF_FLOAT32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_UINT64 = 4, 6, 7, 8, 9, 10
GGML_F32, GGML_F16, GGML_Q8_0 = 0, 1, 8


def f16_bytes(values):
    return struct.pack("<%de" % len(values), *values)


def q8_0_bytes(values):
    out = bytearray()
    for i in range(0, len(values), 32):
        block = values[i:i + 32]
        amax = max(abs(v) for v in block)
        d = amax / 127.0 if amax > 0 else 0.0
        inv = 1.0 / d if d > 0 else 0.0
        out += struct.pack("<e", d)
        out += struct.pack("<32b", *[max(-127, min(127, int(round(v * inv)))) for v in block])
    return bytes(out)


class Writer:
    def __init__(self):
        self.kv = []
        self.tensors = []

    def add_u32(self, k, v):
        self.kv.append((k, GGUF_UINT32, struct.pack("<I", v)))

    def add_f32(self, k, v):
        self.kv.append((k, GGUF_FLOAT32, struct.pack("<f", v)))

    def add_str(self, k, v):
        b = v.encode()
        self.kv.append((k, GGUF_STRING, struct.pack("<Q", len(b)) + b))

    def add_tensor(self, name, ne, gtype, data):
        self.tensors.append((name, ne, gtype, data))

    def write(self, path):
        align = 32
        body = bytearray()
        body += b"GGUF" + struct.pack("<I", 3) + struct.pack("<q", len(self.tensors)) + struct.pack("<q", len(self.kv))
        for k, t, v in self.kv:
            kb = k.encode()
            body += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", t) + v
        offset = 0
        offsets = []
        for name, ne, gtype, data in self.tensors:
            offsets.append(offset)
            offset += (len(data) + align - 1) // align * align
        for (name, ne, gtype, data), off in zip(self.tensors, offsets):
            nb = name.encode()
            body += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(ne))
            for d in ne:
                body += struct.pack("<Q", d)
            body += struct.pack("<I", gtype) + struct.pack("<Q", off)
        pad = (align - len(body) % align) % align
        body += b"\0" * pad
        for name, ne, gtype, data in self.tensors:
            body += data
            body += b"\0" * ((align - len(data) % align) % align)
        with open(path, "wb") as f:
            f.write(body)


def main():
    ap = argparse.ArgumentParser(description="write a tiny random qwen3moe GGUF for engine tests")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--embd", type=int, default=64)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--kv-heads", type=int, default=1)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--used", type=int, default=2)
    ap.add_argument("--ff-exp", type=int, default=32)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    random.seed(a.seed)
    head_dim = a.embd // a.heads
    kv_dim = head_dim * a.kv_heads

    def gauss(n, std=0.05):
        return [random.gauss(0.0, std) for _ in range(n)]

    w = Writer()
    arch = "qwen3moe"
    w.add_str("general.architecture", arch)
    w.add_str("general.name", "tiny-qwen3moe-test")
    w.add_u32(arch + ".block_count", a.layers)
    w.add_u32(arch + ".context_length", 1024)
    w.add_u32(arch + ".embedding_length", a.embd)
    w.add_u32(arch + ".feed_forward_length", a.ff_exp * a.used)
    w.add_u32(arch + ".attention.head_count", a.heads)
    w.add_u32(arch + ".attention.head_count_kv", a.kv_heads)
    w.add_u32(arch + ".attention.key_length", head_dim)
    w.add_u32(arch + ".attention.value_length", head_dim)
    w.add_f32(arch + ".attention.layer_norm_rms_epsilon", 1e-6)
    w.add_f32(arch + ".rope.freq_base", 10000.0)
    w.add_u32(arch + ".expert_count", a.experts)
    w.add_u32(arch + ".expert_used_count", a.used)
    w.add_u32(arch + ".expert_feed_forward_length", a.ff_exp)

    w.add_tensor("token_embd.weight", [a.embd, a.vocab], GGML_F16, f16_bytes(gauss(a.embd * a.vocab, 0.1)))
    w.add_tensor("output_norm.weight", [a.embd], GGML_F32, struct.pack("<%df" % a.embd, *([1.0] * a.embd)))
    for l in range(a.layers):
        p = "blk.%d." % l
        w.add_tensor(p + "attn_norm.weight", [a.embd], GGML_F32, struct.pack("<%df" % a.embd, *([1.0] * a.embd)))
        w.add_tensor(p + "attn_q.weight", [a.embd, a.embd], GGML_F16, f16_bytes(gauss(a.embd * a.embd)))
        w.add_tensor(p + "attn_k.weight", [a.embd, kv_dim], GGML_F16, f16_bytes(gauss(a.embd * kv_dim)))
        w.add_tensor(p + "attn_v.weight", [a.embd, kv_dim], GGML_F16, f16_bytes(gauss(a.embd * kv_dim)))
        w.add_tensor(p + "attn_output.weight", [a.embd, a.embd], GGML_F16, f16_bytes(gauss(a.embd * a.embd)))
        w.add_tensor(p + "attn_q_norm.weight", [head_dim], GGML_F32, struct.pack("<%df" % head_dim, *([1.0] * head_dim)))
        w.add_tensor(p + "attn_k_norm.weight", [head_dim], GGML_F32, struct.pack("<%df" % head_dim, *([1.0] * head_dim)))
        w.add_tensor(p + "ffn_norm.weight", [a.embd], GGML_F32, struct.pack("<%df" % a.embd, *([1.0] * a.embd)))
        w.add_tensor(p + "ffn_gate_inp.weight", [a.embd, a.experts], GGML_F16, f16_bytes(gauss(a.embd * a.experts, 0.3)))
        w.add_tensor(p + "ffn_up_exps.weight", [a.embd, a.ff_exp, a.experts], GGML_Q8_0, q8_0_bytes(gauss(a.embd * a.ff_exp * a.experts)))
        w.add_tensor(p + "ffn_gate_exps.weight", [a.embd, a.ff_exp, a.experts], GGML_Q8_0, q8_0_bytes(gauss(a.embd * a.ff_exp * a.experts)))
        w.add_tensor(p + "ffn_down_exps.weight", [a.ff_exp, a.embd, a.experts], GGML_Q8_0, q8_0_bytes(gauss(a.embd * a.ff_exp * a.experts)))
    w.write(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
