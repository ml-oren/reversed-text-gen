from transformers import BartConfig, BartForConditionalGeneration
from transformers import RobertaModel
import torch

REVERSED_MODEL = True
OUTPUT_DIR = "./models/pretrained_bart_from_roberta-090723-reversed"

bart_config = BartConfig(
    vocab_size=50265,
    d_model=768,
    encoder_ffn_dim=3072,
    encoder_layers=12,
    encoder_attention_heads=12,
    decoder_ffn_dim=3072,
    decoder_layers=12,
    decoder_attention_heads=12,
    max_position_embeddings=512,
)

bart = BartForConditionalGeneration(bart_config)

pretrained_roberta = RobertaModel.from_pretrained("roberta-base")

bart.model.encoder.embed_tokens = pretrained_roberta.embeddings.word_embeddings
bart.model.encoder.embed_positions = pretrained_roberta.embeddings.position_embeddings
for bart_layer, roberta_layer in zip(bart.model.encoder.layers, pretrained_roberta.encoder.layer):
    bart_layer.self_attn.k_proj = roberta_layer.attention.self.key
    bart_layer.self_attn.v_proj = roberta_layer.attention.self.value
    bart_layer.self_attn.q_proj = roberta_layer.attention.self.query
    bart_layer.self_attn.out_proj = roberta_layer.attention.output.dense
    bart_layer.self_attn_layer_norm = roberta_layer.attention.output.LayerNorm
    bart_layer.fc1 = roberta_layer.intermediate.dense
    bart_layer.fc2 = roberta_layer.output.dense
    bart_layer.final_layer_norm = roberta_layer.output.LayerNorm

bart.model.decoder.embed_tokens = pretrained_roberta.embeddings.word_embeddings
bart.model.decoder.embed_positions = pretrained_roberta.embeddings.position_embeddings
for bart_layer, roberta_layer in zip(bart.model.decoder.layers, pretrained_roberta.encoder.layer):
    bart_layer.self_attn.k_proj = roberta_layer.attention.self.key
    bart_layer.self_attn.v_proj = roberta_layer.attention.self.value
    bart_layer.self_attn.q_proj = roberta_layer.attention.self.query
    bart_layer.self_attn.out_proj = roberta_layer.attention.output.dense
    bart_layer.self_attn_layer_norm = roberta_layer.attention.output.LayerNorm
    bart_layer.encoder_attn.k_proj = roberta_layer.attention.self.key
    bart_layer.encoder_attn.v_proj = roberta_layer.attention.self.value
    bart_layer.encoder_attn.q_proj = roberta_layer.attention.self.query
    bart_layer.encoder_attn.out_proj = roberta_layer.attention.output.dense
    bart_layer.encoder_attn_layer_norm = roberta_layer.attention.output.LayerNorm
    bart_layer.fc1 = roberta_layer.intermediate.dense
    bart_layer.fc2 = roberta_layer.output.dense
    bart_layer.final_layer_norm = roberta_layer.output.LayerNorm

bart.model.shared = pretrained_roberta.embeddings.word_embeddings

if REVERSED_MODEL:
    bart.model.decoder.embed_positions.weight = torch.nn.Parameter(bart.model.encoder.embed_positions.weight.flip(0))

bart.save_pretrained(OUTPUT_DIR)