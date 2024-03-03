import dataclasses
import json
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp
from transformers import FlaxGemmaForCausalLM

from gemma_flax.model import GemmaConfig, GemmaModel

def from_hf(name='google/gemma-2b'):
    hf_model, params = FlaxGemmaForCausalLM.from_pretrained(name, _do_init=False, revision='flax')
    out_params = {}

    m_params = params['model']
    out_params['tokens_emb'] = m_params['embed_tokens']

    out_params['norm'] = m_params['norm']

    n_layer = len(m_params['layers'])
    layers =  [m_params['layers'][str(i)] for i in range(n_layer)]

    for layer in layers:
        self_attn = layer['self_attn']
        self_attn['qkv_proj'] = jax.tree_map(
                lambda *xs: jnp.concatenate(xs, axis=-1),
                self_attn['q_proj'], self_attn['k_proj'], self_attn['v_proj'],
        )
        del self_attn['q_proj']
        del self_attn['k_proj']
        del self_attn['v_proj']

    out_params['trunk'] = jax.tree_map(lambda *xs: jnp.stack(xs), *layers)
    out_params = {'params': out_params}

    hf_conf = hf_model.config

    config = GemmaConfig(
        n_vocab=hf_conf.vocab_size,
        hidden_size=hf_conf.hidden_size,
        intermediate_size=hf_conf.intermediate_size,
        num_heads=hf_conf.num_attention_heads,
        num_kv_heads=hf_conf.num_key_value_heads,
        head_dim=hf_conf.head_dim,
        num_layers=hf_conf.num_hidden_layers,
        ln_eps=hf_conf.rms_norm_eps,
        jax_dtype=hf_conf.torch_dtype
    )
    model = GemmaModel(config)
    expected_params = jax.eval_shape(model.init, jax.random.PRNGKey(0), jnp.ones(3, dtype=jnp.int32))
    def check_params(path, a, b):
        assert a.shape == b.shape, f'Shape mismatch at {path}: {a.shape} != {b.shape}'
        assert a.dtype == b.dtype, f'Dtype mismatch at {path}: {a.dtype} != {b.dtype}'

    jax.tree_util.tree_map_with_path(check_params, expected_params, out_params)
    return model, out_params

def save_checkpoint(config_or_model, params, path):
    config = config_or_model if isinstance(config_or_model, GemmaConfig) else config_or_model.config

    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)

    with open(path / 'config.json', 'w') as f:
        json.dump(dataclasses.asdict(config), f)

    with open(path / 'params.pkl', 'wb') as f:
        pickle.dump(jax.device_put(params, jax.devices('cpu')[0]), f)

def load_checkpoint(path):
    path = Path(path)
    with open(path / 'config.json', 'r') as f:
        config = GemmaConfig(**json.load(f))

    with open(path / 'params.pkl', 'rb') as f:
        params = pickle.load(f)

    model = GemmaModel(config)
    return model, params
