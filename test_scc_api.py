import torch

from attack.textAttackSCC import TextAttackerSCC


class DummyTokenizer:
    def __init__(self):
        from transformers import BertTokenizer

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __call__(self, texts, **kwargs):
        return self.tokenizer(texts, **kwargs)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def convert_ids_to_tokens(self, ids):
        return self.tokenizer.convert_ids_to_tokens(ids)


def main():
    print(">>> 正在初始化 Tokenizer 与 SCC 攻击器 (可能需要几秒加载 SBERT) ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DummyTokenizer()

    attacker = TextAttackerSCC(tokenizer=tokenizer, device=device, sim_threshold=0.65)

    test_text = "A group of tourists standing near a white building."
    max_length = 30

    text_inputs = tokenizer(
        [test_text],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    tokens = tokenizer.convert_ids_to_tokens(text_inputs.input_ids[0])

    weights = torch.rand(max_length)

    print("\n" + "=" * 50)
    print(f">>> [阶段 1] 原句: {test_text}")
    print("=" * 50)

    target_chunks = attacker._get_top_n_attention_chunks(
        test_text, tokens, weights, n=2
    )
    print(f"\n>>> [阶段 2] 提取到的高注意力语义块列表: {target_chunks}")

    print("\n>>> [阶段 3] 正在执行混合请求策略 (单点15条 + 双点15条)...")

    raw_replace_texts = []

    if len(target_chunks) >= 2:
        print(f"    -> 正在请求单点替换 (目标: {target_chunks[:1]})...")
        cands_single = attacker._get_llm_substitutes(
            test_text, target_chunks[:1], num_request=15
        )
        raw_replace_texts.extend(cands_single)

        print(f"    -> 正在请求双点替换 (目标: {target_chunks[:2]})...")
        cands_double = attacker._get_llm_substitutes(
            test_text, target_chunks[:2], num_request=15
        )
        raw_replace_texts.extend(cands_double)
    else:
        print("    -> 语义块不足，请求 30 条单点替换...")
        raw_replace_texts = attacker._get_llm_substitutes(
            test_text, target_chunks, num_request=30
        )

    if not raw_replace_texts:
        print("\n[!] 警告: API 未返回任何数据。")
        return

    print(f"\n[API 返回原始混合数据 (总共 {len(raw_replace_texts)} 条)]:")
    for idx, cand in enumerate(raw_replace_texts):
        tag = "[单点]" if idx < 15 else "[双点]"
        print(f"  {idx + 1}. {tag} {cand}")

    print("\n>>> [阶段 4] 正在进行 SBERT 过滤 (阈值 0.65)...")
    final_candidates = attacker._filter_candidates(test_text, raw_replace_texts, k=10)

    print(f"\n[最终通过筛选的候选文本 (Top-{len(final_candidates)})]:")
    for idx, cand in enumerate(final_candidates):
        print(f"  ★ {idx + 1}. {cand}")

    print("\n测试完成。观察 Top 列表，看单点和双点生成的句子是否都能通过 0.65 的阈值。")


if __name__ == "__main__":
    main()
