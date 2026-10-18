import os
import sys
import csv
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm



RINALMO_REPO = Path("./rinalmo")

RINALMO_WEIGHT = Path("./weights/rinalmo_giga_pretrained.pt")

DATA_ROOT = Path("./data")

OUTPUT_ROOT = Path("./features")

SUBSETS = ["A", "U", "C", "G"]

TRAIN_FILE_TEMPLATE = "{subset}_H2Opred_Tr.csv"
TEST_FILE_TEMPLATE = "{subset}_H2Opred_Te.csv"

SEQ_LEN = 41

EXPECTED_DIM = 1280

BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def prepare_rinalmo_source():
    """
    将本地 RiNALMo-main 加入 Python 搜索路径。
    这样即使没有 pip install -e .，也能从源码导入 rinalmo。
    """
    if not RINALMO_REPO.exists():
        raise FileNotFoundError(f"未找到 RiNALMo 代码库目录：{RINALMO_REPO}")

    rinalmo_pkg = RINALMO_REPO / "rinalmo"
    if not rinalmo_pkg.exists():
        raise FileNotFoundError(
            f"在 {RINALMO_REPO} 下没有找到 rinalmo 包目录。\n"
            f"请确认目录结构是否为：{RINALMO_REPO}/rinalmo"
        )

    sys.path.insert(0, str(RINALMO_REPO))


def prepare_rinalmo_weight_link():
    if not RINALMO_WEIGHT.exists():
        raise FileNotFoundError(
            f"未找到 RiNALMo Giga 权重文件：{RINALMO_WEIGHT}\n"
            f"请确认文件是否存在。"
        )

    cache_dir = Path.home() / ".cache" / "rinalmo_pretrained"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_weight = cache_dir / "giga-v1.pt"

    if cache_weight.exists() or cache_weight.is_symlink():
        cache_weight.unlink()

    os.symlink(str(RINALMO_WEIGHT), str(cache_weight))

    print("[Cache] RiNALMo 权重软链接已准备：")
    print(f"        {cache_weight} -> {RINALMO_WEIGHT}")


def check_data_files():
    """
    检查当前配置下的输入 CSV 是否存在。
    """
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"未找到数据目录：{DATA_ROOT}")

    missing_files = []

    for subset in SUBSETS:
        train_path = DATA_ROOT / TRAIN_FILE_TEMPLATE.format(subset=subset)
        test_path = DATA_ROOT / TEST_FILE_TEMPLATE.format(subset=subset)

        if not train_path.exists():
            missing_files.append(str(train_path))
        if not test_path.exists():
            missing_files.append(str(test_path))

    if missing_files:
        raise FileNotFoundError(
            "以下输入 CSV 文件不存在，请检查路径或文件名模板：\n"
            + "\n".join(missing_files)
        )


prepare_rinalmo_source()
prepare_rinalmo_weight_link()
check_data_files()

try:
    from rinalmo.pretrained import get_pretrained_model
except Exception as e:
    raise RuntimeError(
        "导入 RiNALMo 失败。\n\n"
        "请确认：\n"
        "1. 当前环境为 conda activate ENV\n"
        "2. flash-attn 已成功安装\n"
        "3. RiNALMo-main 路径正确\n\n"
        f"原始错误：{repr(e)}"
    )



def pad_or_truncate_seq(seq: str, seq_len: int = 41) -> str:
    """
    统一序列长度为 41 nt。
    - 转大写
    - T 替换为 U
    - 超过 41 nt 截断
    - 不足 41 nt 用 N 补齐
    """
    seq = (seq or "").strip().upper().replace("T", "U")

    if len(seq) >= seq_len:
        return seq[:seq_len]

    return seq + ("N" * (seq_len - len(seq)))


def is_valid_rna_seq(seq: str) -> bool:
    """
    过滤表头或异常行。
    正常 RNA 序列允许 A/U/C/G/N。
    """
    if not seq:
        return False

    return set(seq).issubset(set("AUCGN"))


def read_unique_sequences(csv_paths, seq_len: int = 41):
    """
    从 train/test CSV 的第一列读取 RNA 序列。
    对每个子集内部的训练集和测试集序列合并去重。

    注意：
    这里只读取第一列序列，不读取标签。
    """
    unique_sequences = set()

    for path in csv_paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            for line in reader:
                if len(line) < 1:
                    continue

                raw_seq = line[0].strip()
                seq_use = pad_or_truncate_seq(raw_seq, seq_len)

                if not is_valid_rna_seq(seq_use):
                    continue

                unique_sequences.add(seq_use)

    return sorted(unique_sequences)


def batched_list(items, batch_size: int):
    """
    将序列列表按 batch_size 切分。
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def safe_npy_path(output_dir: Path, seq: str) -> Path:
    """
    以 RNA 序列作为 npy 文件名。
    A/U/C/G/N 对文件名是安全的。
    """
    return output_dir / f"{seq}.npy"


def count_existing_features(output_dir: Path):
    """
    统计输出目录中已有的 .npy 文件数量。
    """
    if not output_dir.exists():
        return 0
    return len(list(output_dir.glob("*.npy")))



def extract_features_pipeline():
    print("=" * 90)
    print("[Init] RiNALMo Giga 离线特征提取")
    print("=" * 90)
    print(f"[Device] {DEVICE}")
    print(f"[Repo] {RINALMO_REPO}")
    print(f"[Weight] {RINALMO_WEIGHT}")
    print(f"[Data] {DATA_ROOT}")
    print(f"[Output] {OUTPUT_ROOT}")
    print(f"[Batch size] {BATCH_SIZE}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n[Model] 正在加载 RiNALMo Giga 650M ...")

    model, alphabet = get_pretrained_model(model_name="giga-v1")

    model = model.to(DEVICE)
    model.eval()

    print("[Model] RiNALMo Giga 加载完成。")

    for subset in SUBSETS:
        print("\n" + "=" * 90)
        print(f"正在处理子集 [{subset}]")
        print("=" * 90)

        output_dir = OUTPUT_ROOT / f"rinalmo_features_H2Opred_{subset}"
        output_dir.mkdir(parents=True, exist_ok=True)

        train_path = DATA_ROOT / TRAIN_FILE_TEMPLATE.format(subset=subset)
        test_path = DATA_ROOT / TEST_FILE_TEMPLATE.format(subset=subset)

        csv_paths = [train_path, test_path]

        print(f"[Input] Train: {train_path}")
        print(f"[Input] Test : {test_path}")

        unique_sequences = read_unique_sequences(csv_paths, SEQ_LEN)

        print(f"📊 子集 [{subset}] 去重后唯一序列数：{len(unique_sequences)}")
        print(f"📁 输出目录：{output_dir}")
        print(f"📦 当前输出目录已有 npy：{count_existing_features(output_dir)}")

        if len(unique_sequences) == 0:
            print(f"⚠️ 子集 [{subset}] 未读取到有效序列，跳过。")
            continue

        todo_sequences = []
        count_skipped = 0

        for seq_use in unique_sequences:
            npy_path = safe_npy_path(output_dir, seq_use)

            if npy_path.exists():
                count_skipped += 1
            else:
                todo_sequences.append(seq_use)

        print(f"⏭️ 已存在并跳过：{count_skipped}")
        print(f"🚀 本次需要新提取：{len(todo_sequences)}")

        count_saved = 0
        count_failed = 0

        total_batches = (len(todo_sequences) + BATCH_SIZE - 1) // BATCH_SIZE

        with torch.no_grad():
            for batch_seqs in tqdm(
                batched_list(todo_sequences, BATCH_SIZE),
                total=total_batches,
                desc=f"Extracting {subset}",
            ):
                try:
                    tokens = torch.tensor(
                        alphabet.batch_tokenize(batch_seqs),
                        dtype=torch.int64,
                        device=DEVICE,
                    )

                    outputs = model(tokens)

                    reps = outputs["representation"]
                    reps = reps[:, 1:-1, :].detach().float().cpu().numpy()

                    if reps.shape[1] != SEQ_LEN or reps.shape[2] != EXPECTED_DIM:
                        print(
                            f"\n❌ 批量维度异常：{reps.shape}，"
                            f"期望 [B, {SEQ_LEN}, {EXPECTED_DIM}]，本批跳过。"
                        )
                        count_failed += len(batch_seqs)
                        continue

                    for seq_use, feat in zip(batch_seqs, reps):
                        if feat.shape != (SEQ_LEN, EXPECTED_DIM):
                            print(f"\n❌ 序列 {seq_use} 特征维度异常：{feat.shape}，跳过。")
                            count_failed += 1
                            continue

                        npy_path = safe_npy_path(output_dir, seq_use)
                        np.save(npy_path, feat)
                        count_saved += 1

                except torch.cuda.OutOfMemoryError as e:
                    print(f"\n❌ CUDA OOM：{repr(e)}")
                    print("建议把脚本顶部 BATCH_SIZE 从 8 改成 4、2 或 1，然后重跑。")
                    torch.cuda.empty_cache()
                    count_failed += len(batch_seqs)
                    continue

                except RuntimeError as e:
                    print(f"\n❌ 子集 [{subset}] 某个 batch 提取失败：{repr(e)}")
                    print("如果是显存不足，请降低 BATCH_SIZE。")
                    count_failed += len(batch_seqs)
                    continue

        print(
            f"✨ 子集 [{subset}] 完成："
            f"新生成 {count_saved}；"
            f"跳过已有 {count_skipped}；"
            f"失败 {count_failed}。"
        )

    print("\n🎉 全部子集 RiNALMo 特征提取完成。")



if __name__ == "__main__":
    extract_features_pipeline()
