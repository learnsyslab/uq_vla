from __future__ import annotations

import torch
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

VLM_ENTROPY_METHOD = "vlm_token_entropy"
VLM_PERPLEXITY_METHOD = "vlm_perplexity"


class VLMTokenSampler:
    """VLM-based token-level uncertainty from the prefix forward pass (no action sampling).

    vlm_token_entropy: average token entropy over all valid prefix positions.
    vlm_perplexity:    exp(-mean log P(lang_token | context)) over language tokens.
    """

    def __init__(self, adapter, method: str):
        self._adapter = adapter
        self._method = method

    @torch.no_grad()
    def conditional_sample_with_uncertainty_batch(
        self, observation: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor]:
        model = self._adapter.model
        lang_tokens = observation[OBS_LANGUAGE_TOKENS]   # (B, T_lang)
        lang_masks = observation[OBS_LANGUAGE_ATTENTION_MASK]  # (B, T_lang)

        images, img_masks = model.prepare_images(observation)
        state = model.prepare_state(observation)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state,
        )
        att_2d = make_att_2d_masks(pad_masks=prefix_pad_masks, att_masks=prefix_att_masks)
        pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        outputs_embeds, _ = model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        vlm_hidden = outputs_embeds[0]  # (B, seq_len, hidden_dim)
        lm_head = model.vlm_with_expert.vlm.lm_head
        log_probs = torch.nn.functional.log_softmax(lm_head(vlm_hidden).float(), dim=-1)

        B, seq_len, _ = log_probs.shape
        T_lang = lang_tokens.shape[1]

        if self._method == VLM_ENTROPY_METHOD:
            valid = prefix_pad_masks.float()
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # (B, seq_len)
            uncertainty = (entropy * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            # Language positions: [seq_len-1-T_lang, seq_len-1); state token is last.
            # logits[t] predicts token t+1, so logits[lang_start-1:lang_start+T_lang-1]
            # predicts lang_tokens.
            lang_start = seq_len - 1 - T_lang
            pred_log_probs = log_probs[:, lang_start - 1 : lang_start + T_lang - 1, :]
            gathered = pred_log_probs.gather(-1, lang_tokens.long().unsqueeze(-1)).squeeze(-1)
            valid_lang = lang_masks.float()
            mean_lnlp = (gathered * valid_lang).sum(dim=1) / valid_lang.sum(dim=1).clamp(min=1)
            uncertainty = (-mean_lnlp).exp()  # perplexity

        dummy_actions = torch.zeros(B, self._adapter.horizon, self._adapter.action_dim)
        return dummy_actions, uncertainty.cpu().float()
