"""Instrumented CPU/fp32 Llama-family teaching engine (Transformers 4.46.3).

All public model operations are serialized by main.py. Never serve this engine
concurrently without its request lock: diagnostic hooks temporarily affect the
shared model. No model weights are changed. No pickle caches are loaded.
"""
import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from math_core import affine_apply, affine_fit

APP_VERSION = "2.1.0"
PROMPT_LIMIT = 256
ATTENTION_LIMIT = 64


def tensor_output(output):
    return output[0] if isinstance(output, tuple) else output


def error_check(name, actual, expected, atol=2e-4, rtol=2e-4):
    a, b = torch.as_tensor(actual), torch.as_tensor(expected)
    error = float((a-b).abs().max())
    passed = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    return {"name": name, "passed": passed, "max_abs_error": error, "atol": atol, "rtol": rtol}


def distribution_metrics(base, changed):
    lp, lq = torch.log_softmax(base.double(), -1), torch.log_softmax(changed.double(), -1)
    p, q = lp.exp(), lq.exp()
    lm = torch.logaddexp(lp, lq)-np.log(2)
    return {"tv": float((p-q).abs().sum()/2),
            "js_nats": float(((p*(lp-lm)).sum()+(q*(lq-lm)).sum())/2),
            "kl_base_to_changed_nats": float((p*(lp-lq)).sum())}


class Engine:
    def __init__(self, model=None, tokenizer=None):
        self.checkpoint = os.getenv("LLM_LAB_MODEL", "HuggingFaceTB/SmolLM2-135M")
        self.revision = os.getenv("LLM_LAB_REVISION", "main")
        self.test_fixture = model is not None
        torch.set_num_threads(max(1, min(8, int(os.getenv("LLM_LAB_THREADS", "4")))))
        if model is None:
            options = {"revision": self.revision, "local_files_only": os.getenv("LLM_LAB_OFFLINE") == "1",
                       "trust_remote_code": False}
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, **options)
            model = AutoModelForCausalLM.from_pretrained(
                self.checkpoint, torch_dtype=torch.float32, attn_implementation="eager",
                use_safetensors=True, **options)
        self.model, self.tokenizer = model.cpu().float().eval(), tokenizer
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        c = self.model.config
        if c.model_type != "llama":
            raise ValueError("This instrument supports Llama-family models only")
        self.layers, self.width, self.vocab = c.num_hidden_layers, c.hidden_size, c.vocab_size
        self.heads, self.kv_heads = c.num_attention_heads, c.num_key_value_heads
        self.head_dim = getattr(c, "head_dim", None) or self.width//self.heads
        self.group_size = self.heads//self.kv_heads
        if self.heads % self.kv_heads or self.heads*self.head_dim != self.width:
            raise ValueError("Unsupported head geometry")
        self.probe = None
        self.probe_report = None
        self.probe_id = None
        config_text = json.dumps(c.to_dict(), sort_keys=True, default=str)
        tok_text = (tokenizer.backend_tokenizer.to_str() if hasattr(tokenizer, "backend_tokenizer")
                    else json.dumps(tokenizer.get_vocab(), sort_keys=True))
        self.provenance = {
            "app_version": APP_VERSION, "checkpoint": self.checkpoint,
            "model_kind": "UNTRAINED TEST FIXTURE" if self.test_fixture else
                ("instruction-tuned" if "instruct" in self.checkpoint.lower() else "base or user-supplied checkpoint"),
            "requested_revision": self.revision, "resolved_revision": getattr(c, "_commit_hash", None),
            "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
            "tokenizer_sha256": hashlib.sha256(tok_text.encode()).hexdigest(),
            "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
            "python": platform.python_version(), "platform": platform.platform(),
            "device": "cpu", "dtype": str(next(model.parameters()).dtype), "attention_backend": "eager",
            "threads": torch.get_num_threads(), "parameter_count": sum(p.numel() for p in model.parameters()),
            "layers": self.layers, "width": self.width, "vocabulary": self.vocab,
            "query_heads": self.heads, "kv_heads": self.kv_heads, "head_dim": self.head_dim,
            "mlp_width": c.intermediate_size, "rms_norm_epsilon": c.rms_norm_eps,
            "rope_theta": c.rope_theta, "max_position_embeddings": c.max_position_embeddings,
            "tied_weights_config": c.tie_word_embeddings,
            "tied_weights_actual": model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(),
            "bytes_per_cache_element": next(model.parameters()).element_size(),
            "kv_bytes_per_token": self.layers*self.kv_heads*self.head_dim*2*next(model.parameters()).element_size(),
            "prompt_limit": PROMPT_LIMIT, "attention_limit": ATTENTION_LIMIT,
            "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
            "add_special_tokens": False, "source_sha256": {},
        }
        root = Path(__file__).resolve().parent
        for name in ("engine.py", "main.py", "math_core.py", "static/index.html", "static/app.js", "static/math.js", "static/style.css"):
            path = root/name
            if path.exists():
                self.provenance["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def tokens(self, ids):
        return [{"position": i, "id": int(t), "text": self.tokenizer.decode([int(t)]),
                 "piece": self.tokenizer.convert_ids_to_tokens(int(t)),
                 "special": int(t) in self.tokenizer.all_special_ids} for i, t in enumerate(ids)]

    def encode(self, prompt, limit=PROMPT_LIMIT):
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            raise ValueError("Enter text that produces at least one token")
        if len(ids) > limit:
            raise ValueError(f"This operation accepts at most {limit} tokens; your prompt has {len(ids)}. Nothing was truncated.")
        if len(ids) > self.model.config.max_position_embeddings:
            raise ValueError("Prompt exceeds the model's configured context")
        return torch.tensor([ids], dtype=torch.long)

    def envelope(self, prompt, ids, operation, **settings):
        return {"run_id": uuid.uuid4().hex, "created_utc": datetime.now(timezone.utc).isoformat(),
                "operation": operation, "prompt": prompt, "tokens": self.tokens(ids[0].tolist()),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "settings": settings, "provenance": self.provenance}

    def describe(self, logits, k=8, temperature=1):
        order = torch.argsort(logits, descending=True, stable=True)
        raw = torch.softmax(logits.double(), -1)
        sample = self.sample_probs(logits, temperature)
        return [{"id": int(t), "text": self.tokenizer.decode([int(t)]), "logit": float(logits[t]),
                 "model_probability": float(raw[t]), "sampling_probability": float(sample[t])}
                for t in order[:k]]

    @staticmethod
    def sample_probs(logits, temperature):
        if temperature < 0 or not np.isfinite(temperature):
            raise ValueError("Temperature must be finite and nonnegative")
        if temperature == 0:
            out = torch.zeros_like(logits, dtype=torch.float64)
            out[int(logits.argmax())] = 1
            return out
        return torch.softmax(logits.double()/temperature, -1)

    @torch.no_grad()
    def predict(self, prompt):
        ids = self.encode(prompt)
        z = self.model(ids, use_cache=False).logits[0, -1]
        return {**self.envelope(prompt, ids, "predict", probability_temperature=1),
                "logits": z.tolist(), "named_tokens": self.describe(z, min(200, self.vocab)),
                "vocab_size": self.vocab, "serialization": "Full float32 values; rounding is display-only"}

    @torch.no_grad()
    def generate(self, prompt, n_tokens=12, temperature=1.0, runs=1, seed=None, use_cache=False, stop_eos=True):
        initial = self.encode(prompt)
        if initial.shape[1]+n_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Prompt plus generation budget exceeds model context")
        results, call_count = [], 0
        # A local generator avoids mutating the process-wide RNG. The entire request
        # is reproducible, while individual runs use base_seed + run_index.
        base_seed = secrets.randbits(63) if seed is None else seed
        for run_index in range(runs):
            effective_seed = (base_seed+run_index) % (2**63-1)
            rng = torch.Generator(device="cpu").manual_seed(effective_seed)
            ids, past, generated, steps = initial.clone(), None, [], []
            started = time.perf_counter()
            for step in range(n_tokens):
                input_ids = ids[:, -1:] if use_cache and past is not None else ids
                output = self.model(input_ids, past_key_values=past, use_cache=use_cache)
                past = output.past_key_values if use_cache else None
                z = output.logits[0, -1]
                raw, sample = torch.softmax(z.double(), -1), self.sample_probs(z, temperature)
                chosen = int(z.argmax()) if temperature == 0 else int(torch.multinomial(sample, 1, generator=rng))
                call_count += 1
                if run_index == 0:
                    candidates = self.describe(z, 5, temperature)
                    order = torch.argsort(z, descending=True, stable=True)
                    steps.append({"step": step+1, "context": self.tokenizer.decode(ids[0].tolist()),
                                  "context_ids": ids[0].tolist(), "context_length": ids.shape[1],
                                  "processed_positions": input_ids.shape[1], "candidates": candidates,
                                  "chosen": {"id": chosen, "text": self.tokenizer.decode([chosen]),
                                             "model_probability": float(raw[chosen]), "sampling_probability": float(sample[chosen]),
                                             "rank": int((order == chosen).nonzero()[0, 0])+1}})
                generated.append(chosen)
                ids = torch.cat([ids, torch.tensor([[chosen]])], -1)
                if stop_eos and chosen == self.tokenizer.eos_token_id:
                    break
            results.append({"seed": effective_seed, "generated_ids": generated,
                            "generated_text": self.tokenizer.decode(generated),
                            "full_text": self.tokenizer.decode(ids[0].tolist()), "steps": steps,
                            "stop_reason": "eos" if stop_eos and generated[-1] == self.tokenizer.eos_token_id else "token_budget",
                            "seconds": time.perf_counter()-started})
        return {**self.envelope(prompt, initial, "generate", temperature=temperature, n_tokens=n_tokens,
                                runs=runs, seed=seed, base_seed=base_seed, use_cache=use_cache, stop_eos=stop_eos),
                "results": results, "forward_passes": call_count,
                "unique_outputs": len({tuple(r["generated_ids"]) for r in results})}

    def capture(self, ids, attention=False):
        states, attn_writes, mlp_writes, handles = {}, {}, {}, []
        for i, layer in enumerate(self.model.model.layers):
            def pre(mod, args, index=i):
                states[index] = args[0].detach().clone()
            def post(mod, args, output, index=i):
                states[index+1] = tensor_output(output).detach().clone()
            def aw(mod, args, output, index=i):
                attn_writes[index] = tensor_output(output).detach().clone()
            def mw(mod, args, output, index=i):
                mlp_writes[index] = output.detach().clone()
            handles.extend([layer.register_forward_pre_hook(pre), layer.register_forward_hook(post),
                            layer.self_attn.register_forward_hook(aw), layer.mlp.register_forward_hook(mw)])
        try:
            output = self.model(ids, use_cache=False, output_attentions=attention)
        finally:
            for handle in handles:
                handle.remove()
        return output, states, attn_writes, mlp_writes

    @torch.no_grad()
    def attention(self, prompt):
        ids = self.encode(prompt, ATTENTION_LIMIT)
        out = self.model(ids, use_cache=False, output_attentions=True)
        if not out.attentions:
            raise ValueError("No attention weights returned; the eager backend is required")
        return {**self.envelope(prompt, ids, "attention", head_indexing="1-based", scale="raw probabilities"),
                "attentions": [a[0].tolist() for a in out.attentions], "n_heads": self.heads,
                "n_layers": self.layers, "n_kv_heads": self.kv_heads,
                "checks": [{"layer": i+1, "row_sum_max_error": float((a.sum(-1)-1).abs().max()),
                            "masked_max": float(torch.triu(a, diagonal=1).abs().max())}
                           for i, a in enumerate(out.attentions)]}

    def manual_block(self, incoming, index):
        block = self.model.model.layers[index]
        attn = block.self_attn
        x = block.input_layernorm(incoming)
        b, n, _ = x.shape
        q = attn.q_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        positions = torch.arange(n).unsqueeze(0)
        cos, sin = self.model.model.rotary_emb(x, positions)
        qr, kr = apply_rotary_pos_emb(q, k, cos, sin)
        expanded_k = kr.repeat_interleave(self.group_size, dim=1)
        expanded_v = v.repeat_interleave(self.group_size, dim=1)
        dots = qr @ expanded_k.transpose(-1, -2)
        scores = dots/(self.head_dim**0.5)
        allowed = torch.ones(n, n, dtype=torch.bool).tril()
        masked = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        weights = torch.softmax(masked.float(), -1).to(x.dtype)
        head_outputs = weights @ expanded_v
        concat = head_outputs.transpose(1, 2).contiguous().reshape(b, n, self.width)
        write = attn.o_proj(concat)
        after_attn = incoming+write
        mlp_input = block.post_attention_layernorm(after_attn)
        gate, up = block.mlp.gate_proj(mlp_input), block.mlp.up_proj(mlp_input)
        activated = block.mlp.act_fn(gate)
        product = activated*up
        mlp_write = block.mlp.down_proj(product)
        return dict(incoming=incoming, normalized=x, q=q, k=k, v=v, qr=qr, kr=kr,
                    rope_cos=cos, rope_sin=sin, dots=dots, scores=scores, weights=weights,
                    head_outputs=head_outputs, concat=concat, attention_write=write, after_attention=after_attn,
                    mlp_input=mlp_input, gate=gate, up=up, activated_gate=activated, gated_product=product,
                    mlp_write=mlp_write, outgoing=after_attn+mlp_write)

    @torch.no_grad()
    def inspect(self, prompt, layer, head, position=None, target_id=None):
        ids = self.encode(prompt, ATTENTION_LIMIT)
        n = ids.shape[1]
        pos = n-1 if position is None else position
        if not 1 <= layer <= self.layers or not 1 <= head <= self.heads or not 0 <= pos < n:
            raise ValueError("Layer, head, or query position is out of range")
        out, states, aw, mw = self.capture(ids, attention=True)
        index, h = layer-1, head-1
        kv = h//self.group_size
        block = self.model.model.layers[index]
        m = self.manual_block(states[index], index)
        selected_weights = m["weights"][0, h, pos]
        weighted_values = selected_weights[:, None]*m["v"][0, kv]
        head_slice = block.self_attn.o_proj.weight[:, h*self.head_dim:(h+1)*self.head_dim]
        contributions = weighted_values @ head_slice.T
        final_raw = states[self.layers][0, pos]
        final = self.model.model.norm(final_raw)
        logits = out.logits[0, pos]
        target = int(logits.argmax()) if target_id is None else target_id
        if not 0 <= target < self.vocab:
            raise ValueError("Target vocabulary ID is out of range")
        unembed = self.model.lm_head.weight[target]
        normalized = m["normalized"][0, pos]
        q_weight = block.self_attn.q_proj.weight[h*self.head_dim]
        def vec(a):
            a = a.detach()
            return {"shape": list(a.shape), "l2_norm": float(a.norm()), "values": a.tolist()}
        stages = {name: vec(m[name][0, pos]) for name in
                  ["incoming", "normalized", "attention_write", "after_attention", "mlp_input",
                   "gate", "up", "activated_gate", "gated_product", "mlp_write", "outgoing"]}
        qbias = block.self_attn.q_proj.bias
        checks = [error_check("Manual attention weights vs eager model", m["weights"], out.attentions[index]),
                  error_check("Manual attention output projection vs model", m["attention_write"], aw[index]),
                  error_check("Manual gated MLP vs model", m["mlp_write"], mw[index]),
                  error_check("Complete reconstructed block vs model", m["outgoing"], states[index+1]),
                  error_check("Final norm + unembedding vs model logits", self.model.lm_head(final), logits),
                  error_check("Weighted values sum vs head output", weighted_values.sum(0), m["head_outputs"][0,h,pos])]
        return {**self.envelope(prompt, ids, "inspect", layer=layer, head=head, position=pos, target_id=target),
                "query_head": head, "shared_kv_head": kv+1, "head_dim": self.head_dim,
                "scale_divisor": self.head_dim**0.5, "vectors": stages,
                "query_before_rope": vec(m["q"][0,h,pos]), "query_after_rope": vec(m["qr"][0,h,pos]),
                "keys_before_rope": vec(m["k"][0,kv]), "keys_after_rope": vec(m["kr"][0,kv]),
                "values": vec(m["v"][0,kv]), "rope_cos": vec(m["rope_cos"][0,pos]),
                "rope_sin": vec(m["rope_sin"][0,pos]),
                "q_coordinate_0_projection": {"weights": q_weight.tolist(), "input": normalized.tolist(),
                    "products": (q_weight*normalized).tolist(), "bias": float(qbias[h*self.head_dim]) if qbias is not None else 0,
                    "result": float(m["q"][0,h,pos,0])},
                "attention_row": [{"position": j, "token": self.tokenizer.decode([int(ids[0,j])]),
                                   "dot_product": float(m["dots"][0,h,pos,j]),
                                   "scaled_score": float(m["scores"][0,h,pos,j]),
                                   "masked": j > pos, "masked_score": None if j > pos else float(m["scores"][0,h,pos,j]),
                                   "weight": float(selected_weights[j]), "multiple_of_uniform": float(selected_weights[j])*(pos+1),
                                   "value_norm": float(m["v"][0,kv,j].norm()),
                                   "weighted_value_norm": float(weighted_values[j].norm()),
                                   "projected_contribution_norm": float(contributions[j].norm())} for j in range(n)],
                "weighted_values": vec(weighted_values), "head_output": vec(m["head_outputs"][0,h,pos]),
                "projected_position_contributions": vec(contributions), "head_residual_write": vec(contributions.sum(0)),
                "unembedding": {"target_id": target, "target_text": self.tokenizer.decode([target]),
                    "raw_final_state": vec(final_raw), "normalized_final_state": vec(final),
                    "weight_row": vec(unembed), "dot_terms": vec(final*unembed),
                    "dot_product": float(final @ unembed), "model_logit": float(logits[target]),
                    "cosine": float(torch.nn.functional.cosine_similarity(final, unembed, dim=0)),
                    "probability_T1": float(torch.softmax(logits.double(),-1)[target])}, "checks": checks}

    @torch.no_grad()
    def lenses(self, prompt, k=3):
        ids = self.encode(prompt)
        out, states, _, _ = self.capture(ids)
        true = out.logits[0,-1]
        target = int(true.argmax())
        stages = []
        for i in range(self.layers+1):
            h = states[i][0,-1]
            z = self.model.lm_head(self.model.model.norm(h))
            entry = {"index": i, "label": "Embedding (raw)" if i == 0 else f"Block {i} output (raw)",
                     "vector_norm": float(h.norm()), "top": self.describe(z, k),
                     "target_rank": int((torch.argsort(z,descending=True,stable=True)==target).nonzero()[0,0])+1,
                     "target_probability": float(torch.softmax(z.double(),-1)[target]),
                     "metrics_to_final": distribution_metrics(true,z), "probe": None}
            if self.probe is not None:
                prediction = affine_apply(h.numpy(), self.probe[i])
                pz = self.model.lm_head(self.model.model.norm(torch.tensor(prediction, dtype=h.dtype)))
                entry["probe"] = {"top": self.describe(pz,k), "metrics_to_final": distribution_metrics(true,pz)}
            stages.append(entry)
        post = self.model.model.norm(states[self.layers][0,-1])
        stages.append({"index": self.layers+1, "label": "Final RMSNorm (already normalized)",
                       "vector_norm": float(post.norm()), "top": self.describe(self.model.lm_head(post),k),
                       "target_rank": 1, "target_probability": float(torch.softmax(true.double(),-1)[target]),
                       "metrics_to_final": distribution_metrics(true,self.model.lm_head(post)), "probe": None})
        return {**self.envelope(prompt,ids,"lenses",probability_temperature=1,probe_id=self.probe_id),
                "target_id":target,"target_text":self.tokenizer.decode([target]),"stages":stages,
                "checks":[error_check("Raw final state, normalized once, reproduces logits",self.model.lm_head(post),true)],
                "probe_report":self.probe_report,"legacy_jacobian_cache_loaded":False}

    def modified_logits(self, ids, index, heads=(), scope="last", uniform=False):
        attn = self.model.model.layers[index].self_attn
        # The pre-hook alters concatenated AV outputs, not parameters or Q/K.
        def hook(module,args):
            x = args[0].clone()
            positions = slice(None) if scope == "all" else slice(-1,None)
            if uniform:
                # Capture of projected V is filled earlier in this same pass.
                v = saved["v"].view(1,ids.shape[1],self.kv_heads,self.head_dim).transpose(1,2)
                uv = v.cumsum(2)/torch.arange(1,ids.shape[1]+1).view(1,1,-1,1)
                for h in heads:
                    x[:,positions,h*self.head_dim:(h+1)*self.head_dim] = uv[:,h//self.group_size,positions,:]
            else:
                for h in heads:
                    x[:,positions,h*self.head_dim:(h+1)*self.head_dim] = 0
            return (x,)+args[1:]
        saved,handles = {},[]
        if uniform:
            def store_values(module,args,output):
                saved["v"] = output
            handles.append(attn.v_proj.register_forward_hook(store_values))
        handles.append(attn.o_proj.register_forward_pre_hook(hook))
        try:
            return self.model(ids,use_cache=False).logits[0,-1]
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def knockout(self,prompt,layer,scope="last",target_id=None,contrast_id=None,uniform=False):
        ids = self.encode(prompt,ATTENTION_LIMIT)
        if not 1 <= layer <= self.layers or scope not in ("last","all"):
            raise ValueError("Invalid intervention layer or scope")
        out = self.model(ids,use_cache=False,output_attentions=True)
        base = out.logits[0,-1]
        order = torch.argsort(base,descending=True,stable=True)
        target = int(order[0]) if target_id is None else target_id
        contrast = next(int(t) for t in order if int(t)!=target) if contrast_id is None else contrast_id
        if not 0 <= target < self.vocab or not 0 <= contrast < self.vocab or target == contrast:
            raise ValueError("Choose two distinct, valid vocabulary IDs")
        bp = torch.softmax(base.double(),-1)
        no_op = self.modified_logits(ids,layer-1,(),scope)
        rows=[]
        for head_indices in [tuple([h]) for h in range(self.heads)]+[tuple(range(self.heads))]:
            z = self.modified_logits(ids,layer-1,head_indices,scope,uniform)
            p = torch.softmax(z.double(),-1)
            rows.append({"head":"all" if len(head_indices)==self.heads else head_indices[0]+1,
                         "sink_share":None if len(head_indices)==self.heads else float(out.attentions[layer-1][0,head_indices[0],-1,0]),
                         **distribution_metrics(base,z), "new_top":self.describe(z,5),
                         "top1_changed":int(z.argmax())!=int(base.argmax()),"target_probability":float(p[target]),
                         "delta_probability_pp":float((p[target]-bp[target])*100),
                         "target_rank":int((torch.argsort(z,descending=True,stable=True)==target).nonzero()[0,0])+1,
                         "logit_difference":float(z[target]-z[contrast]),
                         "delta_logit_difference":float((z[target]-z[contrast])-(base[target]-base[contrast]))})
        return {**self.envelope(prompt,ids,"uniform_intervention" if uniform else "knockout",layer=layer,scope=scope,
                                target_id=target,contrast_id=contrast,probability_temperature=1),
                "target_text":self.tokenizer.decode([target]),"contrast_text":self.tokenizer.decode([contrast]),
                "baseline":{"top":self.describe(base,5),"target_probability":float(bp[target]),
                            "logit_difference":float(base[target]-base[contrast])}, "rows":rows,
                "checks":[error_check("No-op hook reproduces baseline",no_op,base)],"forward_passes":self.heads+3,
                "definition":"TV = 0.5 * sum over full vocabulary of abs(p_baseline - p_intervention). Probabilities use T=1."}

    @torch.no_grad()
    def compare(self,prompt_a,prompt_b,target_ids=None,layer=None,head=1):
        ia,ib = self.encode(prompt_a,ATTENTION_LIMIT),self.encode(prompt_b,ATTENTION_LIMIT)
        oa,sa,_,_ = self.capture(ia,attention=True)
        ob,sb,_,_ = self.capture(ib,attention=True)
        za,zb = oa.logits[0,-1],ob.logits[0,-1]
        pa,pb = torch.softmax(za.double(),-1),torch.softmax(zb.double(),-1)
        selected = list(dict.fromkeys((target_ids or [])+[int(t) for t in za.topk(5).indices]+[int(t) for t in zb.topk(5).indices]))
        if any(not 0 <= t < self.vocab for t in selected):
            raise ValueError("Target ID out of range")
        li = self.layers-1 if layer is None else layer-1
        if not 0 <= li < self.layers or not 1 <= head <= self.heads:
            raise ValueError("Invalid layer/head")
        return {**self.envelope(prompt_a,ia,"compare",prompt_b=prompt_b,layer=li+1,head=head,probability_temperature=1),
                "tokens_b":self.tokens(ib[0].tolist()), "metrics":distribution_metrics(za,zb),
                "candidates":[{"id":t,"text":self.tokenizer.decode([t]),"probability_a":float(pa[t]),
                               "probability_b":float(pb[t]),"delta_pp":float(100*(pb[t]-pa[t])),
                               "logit_a":float(za[t]),"logit_b":float(zb[t])} for t in selected],
                "layers":[{"layer":i,"norm_a":float(sa[i][0,-1].norm()),"norm_b":float(sb[i][0,-1].norm()),
                           "cosine_last_position":float(torch.nn.functional.cosine_similarity(sa[i][0,-1],sb[i][0,-1],dim=0))}
                          for i in range(self.layers+1)],
                "last_row_a":oa.attentions[li][0,head-1,-1].tolist(),"last_row_b":ob.attentions[li][0,head-1,-1].tolist(),
                "warning":"Rows have separate token axes. Different tokenization/length confounds interpretation; cosine is descriptive, not a semantic or causal score."}

    @torch.no_grad()
    def training_trace(self,prompt):
        ids=self.encode(prompt)
        if ids.shape[1]<2:
            raise ValueError("Training trace needs at least two tokens")
        logits=self.model(ids,use_cache=False).logits[0]
        lp=torch.log_softmax(logits[:-1].double(),-1)
        targets=ids[0,1:]
        losses=-lp[torch.arange(len(targets)),targets]
        return {**self.envelope(prompt,ids,"teacher_forcing",weights_updated=False,add_special_tokens=False),
                "rows":[{"position":i,"input_token":self.tokenizer.decode([int(ids[0,i])]),
                         "target_id":int(targets[i]),"target_text":self.tokenizer.decode([int(targets[i])]),
                         "target_probability":float(lp[i,targets[i]].exp()),"loss_nats":float(losses[i]),
                         "d_loss_d_target_logit_unaveraged":float(lp[i,targets[i]].exp()-1)} for i in range(len(targets))],
                "mean_loss_nats":float(losses.mean()),"perplexity":float(losses.mean().exp()),
                "note":"Teacher-forced next-token loss on the provided text, not a factual accuracy test. The last position has no supplied target. No optimizer is run."}

    @torch.no_grad()
    def fit_probe(self,train_prompts,test_prompts,ridge=1.0):
        if set(train_prompts)&set(test_prompts):
            raise ValueError("Calibration and held-out prompts must be disjoint")
        train_ids=[self.encode(p,ATTENTION_LIMIT) for p in train_prompts]
        test_ids=[self.encode(p,ATTENTION_LIMIT) for p in test_prompts]
        train_keys=[tuple(ids[0].tolist()) for ids in train_ids]
        test_keys=[tuple(ids[0].tolist()) for ids in test_ids]
        if len(set(train_keys))!=len(train_keys) or len(set(test_keys))!=len(test_keys) or set(train_keys)&set(test_keys):
            raise ValueError("Duplicate or overlapping token sequences are not allowed")
        def collect(ids_list):
            hs,zs=[],[]
            for ids in ids_list:
                out,states,_,_=self.capture(ids)
                hs.append(np.stack([states[i][0,-1].numpy() for i in range(self.layers+1)]))
                zs.append(out.logits[0,-1])
            return np.stack(hs),zs
        train,_=collect(train_ids)
        test,true_logits=collect(test_ids)
        fits=[affine_fit(train[:,i],train[:,-1],ridge) for i in range(self.layers+1)]
        scores=[]
        for i,fit in enumerate(fits):
            measured=[]
            for j in range(len(test)):
                raw=torch.tensor(test[j,i])
                pred=torch.tensor(affine_apply(test[j,i],fit),dtype=raw.dtype)
                mean=torch.tensor(train[:,-1].mean(0),dtype=raw.dtype)
                def kl(v):
                    return distribution_metrics(true_logits[j],self.model.lm_head(self.model.model.norm(v)))["kl_base_to_changed_nats"]
                measured.append({"heldout_index":j,"raw_lens_kl":kl(raw),"probe_kl":kl(pred),"constant_final_mean_kl":kl(mean),
                                 "state_relative_l2_error":float((pred-torch.tensor(test[j,-1])).norm()/max(float(torch.tensor(test[j,-1]).norm()),1e-12))})
            scores.append({"stage":i,"heldout_examples":measured,
                           **{k:float(np.mean([m[k] for m in measured])) for k in measured[0] if k!="heldout_index"}})
        report={"probe_id":uuid.uuid4().hex,"created_utc":datetime.now(timezone.utc).isoformat(),
                "operation":"fit_affine_probe","method":"Affine residual-state ridge regression with intercept (not KL-trained tuned lens)",
                "train_prompts":train_prompts,"heldout_prompts":test_prompts,"ridge":ridge,"scores":scores,
                "provenance":self.provenance,
                "limitations":"Small, user-visible calibration corpus. Held-out here means not used in fitting, not an independent benchmark. Reusing these examples to tune ridge turns them into validation data; use a fresh test set for final evaluation."}
        # Commit only after all fitting/evaluation has succeeded.
        self.probe,self.probe_report,self.probe_id=fits,report,report["probe_id"]
        return report

    def jacobian(self,prompt,layer,coordinate=0,epsilon=0.01):
        """A local JVP of final pre-norm state wrt last position at block L.

        All earlier positions at the selected block output are held fixed.
        Compare derivative against central finite differences; do not use J*h.
        """
        ids=self.encode(prompt,ATTENTION_LIMIT)
        if not 1 <= layer <= self.layers or not 0 <= coordinate < self.width:
            raise ValueError("Invalid stage or residual coordinate")
        with torch.no_grad():
            out,states,_,_=self.capture(ids)
        raw=states[layer].detach()
        h0=raw[0,-1].clone()
        direction=torch.zeros_like(h0);direction[coordinate]=1
        # Replace the selected raw block output in a full forward pass, preserving
        # prefix representations. Differentiate only wrt the substituted last row.
        def downstream(h):
            replacement=torch.cat([raw[:,:-1],h.reshape(1,1,-1)],dim=1)
            final_store={}
            def replace(module,args,output):
                return (replacement,)+output[1:] if isinstance(output,tuple) else replacement
            def final_hook(module,args,output):
                final_store["value"]=tensor_output(output)[0,-1]
            h1=self.model.model.layers[layer-1].register_forward_hook(replace)
            h2=self.model.model.layers[-1].register_forward_hook(final_hook)
            try:
                self.model(ids,use_cache=False)
                return final_store["value"]
            finally:
                h2.remove();h1.remove()
        with torch.enable_grad():
            base,jv=torch.autograd.functional.jvp(downstream,h0,direction,create_graph=False,strict=False)
        with torch.no_grad():
            plus,minus=downstream(h0+epsilon*direction),downstream(h0-epsilon*direction)
            finite=(plus-minus)/(2*epsilon)
            approx=base+epsilon*jv
        return {**self.envelope(prompt,ids,"local_jacobian",layer=layer,coordinate=coordinate,epsilon=epsilon,
                                mapping="last position at raw block L output -> final pre-norm last position; all earlier positions fixed"),
                "base":base.detach().tolist(),"jvp":jv.detach().tolist(),"finite_difference":finite.tolist(),
                "actual_perturbed":plus.tolist(),"local_prediction":approx.detach().tolist(),
                "jvp_relative_error":float((jv-finite).norm()/finite.norm().clamp_min(1e-12)),
                "approximation_relative_error":float((plus-approx).norm()/plus.norm().clamp_min(1e-12)),
                "checks":[error_check("Intervention anchor matches actual final pre-norm state",base,states[self.layers][0,-1]),
                          error_check("Autograd JVP vs central finite difference",jv,finite,atol=2e-2,rtol=2e-2)],
                "warning":"Finite differences depend on step size and float32 cancellation. A derivative check is not validation of a global predictor."}

    @torch.no_grad()
    def verify(self,prompt):
        ids=self.encode(prompt, min(ATTENTION_LIMIT,16))
        output,states,aw,mw=self.capture(ids,attention=True)
        checks=[]
        for i in range(self.layers):
            m=self.manual_block(states[i],i)
            checks.extend([error_check(f"Block {i+1}: reconstructed weights",m["weights"],output.attentions[i]),
                           error_check(f"Block {i+1}: reconstructed attention write",m["attention_write"],aw[i]),
                           error_check(f"Block {i+1}: reconstructed MLP",m["mlp_write"],mw[i]),
                           error_check(f"Block {i+1}: raw output",m["outgoing"],states[i+1]),
                           error_check(f"Block {i+1}: row sums",output.attentions[i].sum(-1),torch.ones_like(output.attentions[i].sum(-1))),
                           error_check(f"Block {i+1}: future masking",torch.triu(output.attentions[i],diagonal=1),torch.zeros_like(output.attentions[i]),0,0)])
        final=self.model.lm_head(self.model.model.norm(states[self.layers]))
        checks.append(error_check("Final readout normalized exactly once",final,output.logits))
        baseline=output.logits[0,-1]
        checks.append(error_check("No-op diagnostic hook",self.modified_logits(ids,self.layers-1,()),baseline))
        # Incremental cached and uncached prefixes must agree at each position.
        past=None
        for i in range(ids.shape[1]):
            cached=self.model(ids[:,i:i+1],past_key_values=past,use_cache=True)
            past=cached.past_key_values
            prefix=self.model(ids[:,:i+1],use_cache=False).logits[0,-1]
            checks.extend([error_check(f"Position {i}: cached vs prefix",cached.logits[0,-1],prefix,1e-3,2e-4),
                           error_check(f"Position {i}: causal prefix vs full",prefix,output.logits[0,i],1e-3,2e-4)])
        g1=self.generate(prompt,3,0.7,1,123,False,False)
        g2=self.generate(prompt,3,0.7,1,123,False,False)
        checks.append({"name":"Seed replay in this environment","passed":g1["results"][0]["generated_ids"]==g2["results"][0]["generated_ids"]})
        for temp in (0,0.3,1,3):
            checks.append(error_check(f"Sampling probabilities sum to one at T={temp}",self.sample_probs(baseline,temp).sum(),torch.tensor(1,dtype=torch.float64),1e-12,1e-12))
        # Check all-head zeroing against a separate hook replacing the whole
        # attention write with zero (only valid for bias-free output projection).
        def zero_attention(module,args,out):
            value=tensor_output(out).clone();value[:,-1]=0
            return (value,)+out[1:] if isinstance(out,tuple) else value
        if self.model.model.layers[-1].self_attn.o_proj.bias is None:
            handle=self.model.model.layers[-1].self_attn.register_forward_hook(zero_attention)
            try:
                independent=self.model(ids,use_cache=False).logits[0,-1]
            finally:
                handle.remove()
            checks.append(error_check("All-head removal vs zero attention module output",
                                      self.modified_logits(ids,self.layers-1,tuple(range(self.heads))),independent))
        checks.append(error_check("Hooks restored after diagnostics",self.model(ids,use_cache=False).logits[0,-1],baseline))
        return {**self.envelope(prompt,ids,"verify",runtime_tests=True),"checks":checks,
                "passed":all(c["passed"] for c in checks),"passed_count":sum(c["passed"] for c in checks),
                "check_count":len(checks),"scope":"This prompt and environment only; no claim about semantic faithfulness of probes or all prompts."}
