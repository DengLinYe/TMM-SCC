import os

# 排除列表：不希望包含在内的文件夹和文件
EXCLUDE_DIRS = {
    "tmm_env",
    "data",
    "checkpoints",
    ".git",
    "__pycache__",
    "output",
    ".vscode",
}
EXCLUDE_FILES = {"TMM_Codebase.txt", "snapshot.py", ".gitignore"}
# 允许读取的文件后缀
EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".md"}


def generate_snapshot():
    with open("TMM_Codebase.txt", "w", encoding="utf-8") as outfile:
        for root, dirs, files in os.walk("."):
            # 过滤排除目录
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file in files:
                if file in EXCLUDE_FILES:
                    continue

                ext = os.path.splitext(file)[1]
                if ext in EXTENSIONS:
                    file_path = os.path.join(root, file)
                    outfile.write(f"\n{'=' * 20}\n")
                    outfile.write(f"FILE: {file_path}\n")
                    outfile.write(f"{'=' * 20}\n\n")

                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f"ERROR READING FILE: {e}")
                    outfile.write("\n")


if __name__ == "__main__":
    generate_snapshot()
    print("项目代码已导出至 TMM_Codebase.txt")
