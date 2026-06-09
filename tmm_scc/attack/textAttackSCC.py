import concurrent.futures
import json
import os
import re

import spacy
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

filter_words = set(
    [
        "a",
        ".",
        "-",
        "a the",
        "/",
        "?",
        '"',
        ",",
        "b",
        "&",
        "!",
        "@",
        "%",
        "^",
        "*",
        "(",
        ")",
        "-",
        "-",
        "+",
        "=",
        "<",
        ">",
        "|",
        ":",
        ";",
        "～",
        "·",
    ]
)


class TextAttackerSCC:
    def __init__(self, tokenizer, device, cls=True, sim_threshold=0.75):
        self.tokenizer = tokenizer
        self.cls = cls
        self.device = device
        self.cosSimilatity = torch.nn.CosineEmbeddingLoss()

        self.nlp = spacy.load("en_core_web_sm")
        self.sim_model = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)
        self.sim_threshold = sim_threshold

        self.llm_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def attack(
        self,
        net,
        images,
        texts,
        k=10,
        num_perturbation=1,
        threshold_pred_score=0.3,
        max_length=30,
        batch_size=32,
    ):
        text_inputs = self._prepare_text_inputs(texts, max_length)
        origin_output = net.inference(images, text_inputs)
        origin_embeds = self._get_origin_embeds(origin_output)
        adv_texts = self._generate_adv_texts(
            origin_output, texts, origin_embeds, net, images, k, max_length
        )
        return adv_texts

    def _prepare_text_inputs(self, texts, max_length):
        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        return text_inputs

    def _get_origin_embeds(self, origin_output):
        if self.cls:
            origin_embeds = origin_output["fusion_output"][:, 0, :].detach()
        else:
            origin_embeds = origin_output["fusion_output"].flatten(1).detach()
        return origin_embeds

    def _get_top_n_attention_chunks(self, text, tokens, weights, n=2):
        doc = self.nlp(text)
        chunks = [chunk.text for chunk in doc.noun_chunks]
        chunks.extend([token.text for token in doc if token.pos_ == "VERB"])

        chunk_info = []
        for chunk in set(chunks):
            if chunk.lower() in filter_words:
                continue

            chunk_tokens = self.tokenizer.tokenize(chunk)
            if not chunk_tokens:
                continue

            seq_len = len(chunk_tokens)
            for i in range(len(tokens) - seq_len + 1):
                if tokens[i : i + seq_len] == chunk_tokens:
                    score = weights[i : i + seq_len].mean().item()
                    chunk_info.append({"text": chunk, "score": score})
                    break

        chunk_info.sort(key=lambda x: x["score"], reverse=True)

        unique_chunks = []
        seen = set()
        for c in chunk_info:
            if c["text"] not in seen:
                unique_chunks.append(c["text"])
                seen.add(c["text"])
            if len(unique_chunks) >= n:
                break

        if not unique_chunks:
            unique_chunks = [text.split()[0]]

        return unique_chunks

    def _get_llm_substitutes(self, text, target_chunks, num_request):
        targets_str = ", ".join([f"'{c}'" for c in target_chunks])
        length_rules = "\n".join(
            [
                f"   - The replacement for '{c}' MUST be exactly {len(c.split())} words."
                for c in target_chunks
            ]
        )
        full_word_count = len(text.split())

        prompt = (
            f"You are a poet and linguist. Your task is to rewrite the sentence by replacing the phrases {targets_str} with more beautiful and appropriate expressions.\n\n"
            f"CRITICAL CONSTRAINTS:\n"
            f"1. Replacement Action: Delete the original phrases and insert the new target phrases in their exact original positions.\n"
            f"2. Exact Word Count: The new full sentence MUST have EXACTLY {full_word_count} words (counted by spaces). Furthermore:\n"
            f"{length_rules}\n"
            f"3. Core Imagery: You MUST maintain the core visual imagery and respect the original facts of the scene.\n"
            f"4. Creative Diversity: Under constraint 3, find as many completely new combinations of words as possible to describe the imagery.\n"
            f"5. Output Format: Output ONLY a valid JSON object with a single key 'substitutes' containing a list of full sentence strings. Do NOT output any other text.\n"
            f"6. Quantity: Provide exactly {num_request} options.\n\n"
            f"Original sentence: {text}"
        )

        try:
            response = self.llm_client.chat.completions.create(
                model="deepseek-v3",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a master poet specializing in constrained writing. You strictly output JSON format.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content

            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return data.get("substitutes", [])
            return []

        except Exception as e:
            error_msg = str(e)
            if "403" in error_msg or "AllocationQuota.FreeTierOnly" in error_msg:
                print(
                    f"[Warning] LLM API Quota Exceeded or 403 Forbidden. Details: {error_msg}"
                )
            else:
                print(f"[Error] LLM API Error: {error_msg}")
            return []

    def _generate_adv_texts(
        self, origin_output, texts, origin_embeds, net, images, k, max_length
    ):
        adv_texts = [None] * len(texts)

        batch_tasks = []
        for i, text in enumerate(texts):
            weights = origin_output["weights"][i]
            input_ids = self._prepare_text_inputs([text], max_length).input_ids[0]
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            target_chunks = self._get_top_n_attention_chunks(text, tokens, weights, n=2)
            batch_tasks.append((i, text, target_chunks))

        raw_replace_texts_list = [[] for _ in range(len(texts))]

        def fetch_llm(task):
            idx, text, chunks = task
            res = []
            if len(chunks) >= 2:
                cands_single = self._get_llm_substitutes(
                    text, chunks[:1], num_request=5
                )
                cands_double = self._get_llm_substitutes(
                    text, chunks[:2], num_request=5
                )
                res.extend(cands_single)
                res.extend(cands_double)
            else:
                # 退化情况，请求 20 条单点
                res = self._get_llm_substitutes(text, chunks[:1], num_request=10)
            return idx, res

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(fetch_llm, task) for task in batch_tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    idx, res = future.result()
                    raw_replace_texts_list[idx] = res
                except Exception as e:
                    print(f"[并发请求错误] {e}")

        for i, text in enumerate(texts):
            raw_replace_texts = raw_replace_texts_list[i]

            if not raw_replace_texts:
                adv_texts[i] = text
                continue

            replace_texts = self._filter_candidates(text, raw_replace_texts, k)

            if not replace_texts:
                adv_texts[i] = text
                continue

            adv_embeds_list = []
            valid_replace_texts = []

            for rep_text in replace_texts:
                rep_inputs = self._prepare_text_inputs([rep_text], max_length)
                img_input = images[i].unsqueeze(0) if len(images.shape) == 4 else images
                adv_output = net.inference(img_input, rep_inputs)

                adv_embeds = self._get_adv_embeds(adv_output)
                adv_embeds_list.append(adv_embeds[0])
                valid_replace_texts.append(rep_text)

            if not adv_embeds_list:
                adv_texts[i] = text
                continue

            adv_embeds_tensor = torch.stack(adv_embeds_list)
            losses = self._calculate_loss(
                origin_embeds, adv_embeds_tensor, i, len(valid_replace_texts)
            )

            best_idx = torch.argmin(losses).item()
            adv_texts[i] = valid_replace_texts[best_idx]

        return adv_texts

    def _filter_candidates(self, original_text, candidates, k):
        valid_candidates = []
        orig_len = len(original_text.split())
        orig_emb = self.sim_model.encode(original_text, convert_to_tensor=True)

        for cand in candidates:
            if not isinstance(cand, str):
                continue

            cand_len = len(cand.split())
            if cand_len != orig_len:
                continue

            cand_emb = self.sim_model.encode(cand, convert_to_tensor=True)
            sim_score = util.cos_sim(orig_emb, cand_emb).item()

            if sim_score >= self.sim_threshold:
                valid_candidates.append((cand, sim_score))

        valid_candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in valid_candidates][:k]

    def _get_adv_embeds(self, adv_output):
        if self.cls:
            adv_embeds = adv_output["fusion_output"][:, 0, :].detach()
        else:
            adv_embeds = adv_output["fusion_output"].flatten(1).detach()
        return adv_embeds

    def _calculate_loss(self, origin_embeds, adv_embeds, i, len_replace_texts):
        loss = []
        for j in range(len_replace_texts):
            y = torch.ones(1).to(self.device)
            lossss = self.cosSimilatity(
                origin_embeds[i].unsqueeze(0), adv_embeds[j].unsqueeze(0), y
            )
            loss.append(lossss)
        loss = torch.stack(loss)
        return loss
