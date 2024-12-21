from transformers.models.t5.modeling_t5 import T5LayerNorm, T5Config, T5DenseActDense, T5DenseGatedActDense, T5Attention, T5Model, T5Stack, T5_INPUTS_DOCSTRING, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings, DEPARALLELIZE_DOCSTRING, _CONFIG_FOR_DOC, load_tf_weights_in_t5
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.utils import logging
from transformers.file_utils import add_start_docstrings_to_model_forward
# from transformers.models.t5 import T5PreTrainedModel
from transformers import T5Config
from typing import Optional, Tuple, Union
import copy
import torch
from torch import nn
import warnings
from torch.nn import CrossEntropyLoss
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
# import PreTrainedModel
from transformers.modeling_utils import PreTrainedModel
from transformers.models.t5.modeling_t5 import T5PreTrainedModel
from transformers.file_utils import DUMMY_INPUTS, DUMMY_MASK
import torch.nn.functional as F
import wandb


__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

logger = logging.get_logger(__name__)


class VectorQuantizer(nn.Module):
    def __init__(self, embedding, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding = embedding
        self._commitment_cost = commitment_cost
        self.decoder_vocab_size = self.embedding.weight.shape[0]

    @torch.no_grad()
    def compute_distance(self, inputs, codebook=None):
        codebook_t = codebook.t()
        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)  # [B, S, n_embed or n_embed+1]
        return distances

    def forward(self, inputs_embeds, lab_seq_len, attention_mask):
        quant_n_embed = (self.decoder_vocab_size - 2)//(lab_seq_len-1)
        j = 0
        initial_shape = inputs_embeds.shape

        for i in range(0, self.decoder_vocab_size-2, quant_n_embed):
            curr_seq_em = self.embedding.weight[i+2:i+quant_n_embed+2, :]
            curr_dist = self.compute_distance(inputs_embeds, curr_seq_em)
            curr_dist_flat = curr_dist.view(-1, quant_n_embed)
            curr_idx = torch.argmin(curr_dist_flat, dim=1).unsqueeze(1)
            curr_enc = torch.zeros(curr_idx.shape[0], quant_n_embed).to(inputs_embeds.device)
            curr_enc.scatter_(1, curr_idx, 1)

            if i == 0:
                quantized = torch.matmul(curr_enc, curr_seq_em).view(initial_shape)
            else:
                quantized += torch.matmul(curr_enc, curr_seq_em).view(initial_shape)
            
            j += 1.0
            
        quantized = quantized/j

        # masked_quantized = quantized*attention_mask.unsqueeze(-1)
        # masked_inputs_embeds = inputs_embeds*attention_mask.unsqueeze(-1)

        # flat_masked_quantized = masked_quantized.view(-1, quantized.shape[-1])
        # flat_masked_inputs_embeds = masked_inputs_embeds.view(-1, quantized.shape[-1])

        e_latent_loss = F.mse_loss(quantized*attention_mask.unsqueeze(-1), inputs_embeds*attention_mask.unsqueeze(-1))
        q_latent_loss = F.mse_loss(quantized*attention_mask.unsqueeze(-1), inputs_embeds*attention_mask.unsqueeze(-1))
        loss = q_latent_loss + self._commitment_cost * (e_latent_loss)
        wandb.log({"e_latent_loss": e_latent_loss, "q_latent_loss": q_latent_loss, "loss": loss})

        quantized = inputs_embeds + (quantized - inputs_embeds).detach()
        # quantized = quantized.view(initial_shape)

        return quantized, loss

class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        # print(self.lm_head)
        self.model_dim = config.d_model
        self.decoder_vocab_size = getattr(config, "decoder_vocab_size", None)
        custom = getattr(config, "custom", False)
        self.vq = getattr(config, "vq", False)
        self.docid_max_len = getattr(config, "docid_max_len", 0)
        self._commitment_cost = 0.25

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        self.decoder_embed = nn.Embedding(config.decoder_vocab_size, config.d_model)
        # self.decoder_embed_copy = nn.Embedding(config.decoder_vocab_size, config.d_model)

        dec_num_embed = self.decoder_embed.weight.shape[0]

        # self.decoder_embed.weight.data = self.shared.weight[:dec_num_embed, :]
        # self.decoder_embed_copy.weight.data = self.shared.weight[:dec_num_embed, :]

        if self.vq:
            self.super_quantizer = VectorQuantizer(self.decoder_embed, self._commitment_cost)


        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        
        self.decoder = T5Stack(decoder_config, self.decoder_embed)

        self.lm_head = nn.Linear(config.d_model, config.decoder_vocab_size, bias=False)
        
        print(self.lm_head)
        self.post_init()
        # self.init_weights()
        print("post init")
        self._tie_or_clone_weights(self.lm_head, self.decoder_embed)
        # self._tie_or_clone_weights(self.decoder_embed, self.decoder_embed_copy)
        print(self.lm_head)
        # print()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    @torch.no_grad()
    def compute_distance(self, inputs, codebook=None):
        codebook_t = codebook.t()
        (embed_dim, _) = codebook_t.shape
        inputs_shape = inputs.shape
        assert inputs_shape[-1] == embed_dim

        inputs_flat = inputs.reshape(-1, embed_dim)

        inputs_norm_sq = inputs_flat.pow(2.).sum(dim=1, keepdim=True)
        codebook_t_norm_sq = codebook_t.pow(2.).sum(dim=0, keepdim=True)
        distances = torch.addmm(
            inputs_norm_sq + codebook_t_norm_sq,
            inputs_flat,
            codebook_t,
            alpha=-2.0,
        )
        distances = distances.reshape(*inputs_shape[:-1], -1)  # [B, S, n_embed or n_embed+1]
        return distances

    # @torch.no_grad()
    # def quantize(self, input_embed, seq_len):
    #     quant_n_embed = (self.decoder_vocab_size - 2)//(seq_len-1)
    #     j = 0

    #     for i in range(0, self.decoder_vocab_size-2, quant_n_embed):
    #         curr_seq_em = self.decoder_embed_copy.weight[i+2:i+quant_n_embed+2, :]
    #         curr_dist = self.compute_distance(input_embed, curr_seq_em)
    #         curr_dist_flat = curr_dist.view(-1, quant_n_embed)
    #         curr_idx = torch.argmin(curr_dist_flat, dim=1).unsqueeze(1)
    #         curr_enc = torch.zeros(curr_idx.shape[0], quant_n_embed).to(input_embed.device)
    #         curr_enc.scatter_(1, curr_idx, 1)

    #         if i == 0:
    #             quantized = torch.matmul(curr_enc, curr_seq_em).view(input_embed.shape)
    #         else:
    #             quantized += torch.matmul(curr_enc, curr_seq_em).view(input_embed.shape)
            
    #         j += 1.0
            
    #     return quantized/j

    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.decoder_embed)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    
    def get_input_decode_embeddings(self):
        return self.decoder_embed
    
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.

        If the `torchscript` flag is set in the configuration, can't handle parameter sharing so we are cloning the
        weights instead.
        """
        if getattr(self.config, "tie_word_embeddings", True):
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None:
                self._tie_or_clone_weights(output_embeddings, self.get_input_decode_embeddings())

        if getattr(self.config, "is_encoder_decoder", False) and getattr(self.config, "tie_encoder_decoder", False):
            if hasattr(self, self.base_model_prefix):
                self = getattr(self, self.base_model_prefix)
            self._tie_encoder_decoder_weights(self.encoder, self.decoder, self.base_model_prefix)

        for module in self.modules():
            if hasattr(module, "_tie_weights"):
                module._tie_weights()
    
    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        input_text: Optional[str] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # input_ids_ = input_ids
        # if input_text is not None:
        #     print("input_text: ", input_text)

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        
        

        # Encode if needed (training, first prediction pass)
        loss = None
        if encoder_outputs is None:
            if self.vq and inputs_embeds is None:
                inputs_embeds = self.shared(input_ids)
                quantized, loss = self.super_quantizer(inputs_embeds, self.docid_max_len, attention_mask)

                inputs_embeds = quantized
                input_ids = None
            
            elif self.vq and inputs_embeds is not None:
                quantized, loss = self.super_quantizer(inputs_embeds, self.docid_max_len, attention_mask)

                inputs_embeds = quantized
                input_ids = None
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        # print(hidden_states.sum())

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        # loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)

            if loss is not None:
                loss += loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            else:
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)
    

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past
