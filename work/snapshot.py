"""제출 버전을 versions/ 아래에 폴더로 통째로 남긴다.

    python snapshot.py v1 --score 897.09 --note "기본 47피처 + 드리프트 보정"

versions/v1_LB897.09/
    NOTES.md          무엇을 했고 점수가 얼마인지
    submit.zip        실제로 올린 파일
    model/            그때의 가중치
    code/             그때의 train.py / features.py / script.py / requirements.txt
"""
import argparse
import datetime
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
VERSIONS = os.path.join(ROOT, "versions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="버전 이름 (예: v1)")
    ap.add_argument("--score", default=None, help="리더보드 점수 (모르면 생략)")
    ap.add_argument("--holdout", default=None, help="2024 홀드아웃 점수")
    ap.add_argument("--note", default="", help="한 줄 설명")
    args = ap.parse_args()

    folder = args.name if args.score is None else f"{args.name}_LB{args.score}"
    dst = os.path.join(VERSIONS, folder)
    if os.path.exists(dst):
        raise SystemExit(f"이미 있음: {dst}  (다른 이름을 쓰거나 직접 지우세요)")

    os.makedirs(os.path.join(dst, "code"))
    shutil.copytree(os.path.join(ROOT, "submit", "model"), os.path.join(dst, "model"))
    for src, name in [
        (os.path.join(ROOT, "submit", "script.py"), "script.py"),
        (os.path.join(ROOT, "submit", "requirements.txt"), "requirements.txt"),
        (os.path.join(HERE, "train.py"), "train.py"),
        (os.path.join(HERE, "features.py"), "features.py"),
    ]:
        shutil.copy2(src, os.path.join(dst, "code", name))

    zip_path = os.path.join(ROOT, "submit.zip")
    if os.path.exists(zip_path):
        shutil.copy2(zip_path, os.path.join(dst, "submit.zip"))

    with open(os.path.join(dst, "NOTES.md"), "w", encoding="utf-8") as f:
        f.write(f"# {folder}\n\n")
        f.write(f"- 저장 시각: {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"- 리더보드 Public: {args.score or '미제출'}\n")
        f.write(f"- 2024 홀드아웃: {args.holdout or '-'}\n")
        f.write(f"- 설명: {args.note or '-'}\n")

    total = sum(os.path.getsize(os.path.join(r, x))
                for r, _, fs in os.walk(dst) for x in fs)
    print(f"저장 완료: {dst}  ({total / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
