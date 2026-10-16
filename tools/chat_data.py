"""Chat-style training data: Cyrillic chat messages -> informal Latin pairs (texting shorthand).

  python tools/chat_data.py expand   # local Gemma (Ollama) writes more Cyrillic chat lines from the seeds
  python tools/chat_data.py pairs    # seeds + expanded lines -> data/chat/{train,valid}.jsonl
  python tools/chat_data.py test     # data/raw/dummy_chat_*.json -> data/chat_test.jsonl (evaluation only)
  python tools/chat_data.py mix      # stage-3 mix: chat train + real_train x20 + 10k v4 synthetic

Cyrillic sources: data/chat_seed_cyr.txt (hand-written) and data/chat/expanded_cyr.txt (Gemma). The Latin side
reuses generate_pairs.make_pair with every writer in "chat" mode, plus extra shorthand (mdku, tgvl, dr, lda, yvy,
zza ...) and two src-only habits: question particles glued to the word (ireh uu -> irehuu) and dropped hyphens
before suffixes (12:30-д -> 12:30 d). Lines equal to an evaluation sentence are dropped.
"""
import argparse
import glob
import json
import random
import re
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import generate_pairs as gp  # noqa: E402

SEEDS = ROOT / "data" / "chat_seed_cyr.txt"
SEEDS2 = ROOT / "data" / "chat_seed2_cyr.txt"  # targeted at remaining error classes; repeated in training
OUT = ROOT / "data" / "chat"
EXPANDED = OUT / "expanded_cyr.txt"
CYR = re.compile(r"[Ѐ-ӿ]")

EXTRA_ABBREV = {
    "мэдэхгүй": [("mdku", 0.35), ("mdkv", 0.2), ("medku", 0.1), ("mdehgui", 0.15), ("medehgui", 0.2)],
    "тэгвэл": [("tgvl", 0.5), ("tegvel", 0.3), ("tgwl", 0.2)],
    "тэгээд": [("tgd", 0.4), ("tegeed", 0.6)],
    "дээр": [("dr", 0.4), ("deer", 0.6)],
    "л даа": [("lda", 0.4), ("l daa", 0.6)],
    "явъя": [("yvy", 0.4), ("yavy", 0.3), ("yavya", 0.3)],
    "уулзъя": [("uulzy", 0.5), ("uulzya", 0.5)],
    "үзье": [("uzy", 0.3), ("uzye", 0.4), ("vzy", 0.3)],
    "за за": [("zza", 0.4), ("za za", 0.6)],
    "сайн байна уу": [("sn bnu", 0.4), ("sain bnu", 0.3), ("snu", 0.3)],
    "минут": [("minut", 1)],  # "min" always maps to "мин" (the chat_test convention), never to "минут"
    "ок": [("ok", 0.8), ("oki", 0.1), ("okey", 0.1)],
    "юу": [("yu", 0.7), ("yuu", 0.3)],
    "болохгүй": [("blhgui", 0.4), ("bolohgui", 0.6)],
    "хэрэгтэй": [("hrgtei", 0.3), ("heregtei", 0.7)],
    "шиг": [("shg", 0.3), ("shig", 0.7)],
    "өгчих": [("ugch", 0.4), ("ogch", 0.2), ("ugchih", 0.4)],
    "явчих": [("yvch", 0.4), ("yavchih", 0.6)],
    "хийчих": [("hiich", 0.2), ("hiichih", 0.8)],
    # voluntative -ъя / -ье / -ъё: writers drop the sign and often the final vowel
    "явъя": [("yvy", 0.3), ("yavy", 0.2), ("yavya", 0.3), ("yavii", 0.1), ("yvya", 0.1)],
    "уулзъя": [("uulzy", 0.4), ("uulzya", 0.4), ("uulzii", 0.2)],
    "үзье": [("uzy", 0.3), ("uzye", 0.3), ("vzy", 0.2), ("uzii", 0.2)],
    "ярья": [("yarya", 0.4), ("yary", 0.3), ("yariya", 0.3)],
    "авъя": [("avy", 0.4), ("avya", 0.4), ("avii", 0.2)],
    "хийе": [("hiiy", 0.4), ("hiie", 0.3), ("hiy", 0.3)],
    "идье": [("idye", 0.5), ("idy", 0.3), ("idii", 0.2)],
    "очъё": [("ochy", 0.4), ("ochyo", 0.4), ("ochii", 0.2)],
    "тэгье": [("tgy", 0.3), ("tegye", 0.4), ("tegy", 0.3)],
    "шийдье": [("shiidye", 0.5), ("shiidy", 0.5)],
    "хүлээе": [("huleey", 0.5), ("hvleey", 0.3), ("huleee", 0.2)],
    "явцгаая": [("yavtsgaay", 0.5), ("yavtsgaaya", 0.5)],
    "бодъё": [("body", 0.4), ("bodyo", 0.6)],
    "унтъя": [("unty", 0.5), ("untya", 0.5)],
    "тоглоё": [("togloy", 0.5), ("togloyo", 0.5)],
    "хэлчих": [("helch", 0.4), ("helchih", 0.6)],
    "ирчих": [("irch", 0.4), ("irchih", 0.6)],
    "залгачих": [("zalgach", 0.4), ("zalgachih", 0.6)],
    "авчих": [("avch", 0.4), ("avchih", 0.6)],
    "болзи": [("bolzi", 1)],
    "пфф": [("pff", 0.7), ("pfff", 0.3)],
    "хмм": [("hmm", 0.7), ("hm", 0.3)],
    "хөөх": [("huuh", 0.5), ("hooh", 0.3), ("hvvh", 0.2)],
}
# a suffix typed with a space before it (hural tai -> хуралтай): src only
SPLIT_SUFFIX = re.compile(r"(?<=[a-z]{3})(tai|tei|toi|tay|tei|gui|gvi|guy|tei)\b", re.IGNORECASE)
PARTICLE = re.compile(r"(?<=[a-z])\s+(uu|vv|yu|yuu|yy|uu\?|vv\?)(?=\s|$|[?!.,])", re.IGNORECASE)
SUFFIX_HYPHEN = re.compile(r"(?<=[0-9A-Za-z])-(?=[a-z])")
LATIN_WORD = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?![A-Za-z0-9])")

TOPICS = ["ажил, програм хангамж, deploy, server, алдаа засах", "хоол, кафе, хамт хооллох", "уулзалт, цаг тохирох",
          "худалдаа, үнэ, онлайн захиалга, хүргэлт", "сургууль, хичээл, шалгалт, даалгавар", "гэр бүл, гэрийн ажил",
          "тээвэр, такси, автобус, замын түгжрэл", "тоглоом, кино, цуврал, хөгжим", "эрүүл мэнд, өвчин, эмнэлэг",
          "цаг агаар, хувцас", "мөнгө шилжүүлэх, банк, зээл", "найз нөхөд, үдэшлэг, төрсөн өдөр",
          "аялал, хөдөө явах, амралт", "спорт, фитнесс, сагсан бөмбөг", "утас, компьютер, интернэт, апп",
          "оюутны дотуур байр, түрээс", "хурал, тайлан, дарга", "гэр бүлийн чат, ээж аав, дүү"]

SYSTEM = """Чи монгол залуучуудын өдөр тутмын чат мессежийг кириллээр бичдэг.
Дүрэм:
- Мессеж бүр богино (1-12 үг), ярианы хэллэгтэй, жинхэнэ чат шиг.
- Зөв монгол кирилл бичгээр бич. Орос үг, англи өгүүлбэр бүү бич.
- Монголчууд кириллээр бичдэг гадаад үгсийг кириллээр бич: Ок, мин, монитор, лог, файл, код, сайт.
- Техникийн нэр томьёо, брэнд, бүтээгдэхүүний нэрийг латинаар үлдээ: deploy, server, config, PR, iPhone, Zoom.
- Emoji, дугаарлалт бүү ашигла.
- Мессежүүд бие биенээсээ ялгаатай байх."""

SCHEMA = {"type": "object", "properties": {"messages": {"type": "array", "items": {"type": "string"}}},
          "required": ["messages"]}


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def norm(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def eval_sentences() -> set[str]:
    out = set()
    for p in ["data/real_dev.jsonl", "data/real_test.jsonl", "data/chat_test.jsonl"]:
        f = ROOT / p
        if f.exists():
            for ln in f.read_text(encoding="utf-8-sig").splitlines():
                if ln.strip():
                    out.add(norm(json.loads(ln)["tgt"]))
    return out


def good_line(s: str) -> bool:
    if not (2 <= len(s) <= 110) or not CYR.search(s):
        return False
    latin = sum(c.isascii() and c.isalpha() for c in s)
    letters = sum(c.isalpha() for c in s)
    return latin <= 0.4 * letters and not re.search(r"[ыэ]{3,}|\d+\.|^[-*•]", s)


def cmd_expand(args):
    seeds = read_lines(SEEDS)
    OUT.mkdir(parents=True, exist_ok=True)
    have = {norm(s) for s in seeds + read_lines(EXPANDED)}
    rng = random.Random(args.seed)
    added = 0
    with EXPANDED.open("a", encoding="utf-8") as out:
        for r in range(args.rounds):
            for topic in TOPICS:
                shots = "\n".join(rng.sample(seeds, 12))
                prompt = (f"Сэдэв: {topic}\nЖишээ мессежүүд:\n{shots}\n\n"
                          f"Энэ сэдвээр {args.per_call} шинэ чат мессеж бич. JSON: {{\"messages\": [...]}}")
                body = {"model": args.model, "stream": False, "format": SCHEMA,
                        "options": {"temperature": 0.9, "seed": rng.randint(0, 10**6)},
                        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
                req = urllib.request.Request(f"{args.ollama_url}/api/chat", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                try:
                    msgs = json.loads(json.load(urllib.request.urlopen(req, timeout=300))["message"]["content"])
                    msgs = msgs.get("messages", [])
                except Exception as e:  # noqa: BLE001 - keep going on one bad reply
                    print(f"round {r} {topic}: {e}", flush=True)
                    continue
                for m in msgs:
                    m = re.sub(r"\s+", " ", str(m)).strip().strip('"')
                    if good_line(m) and norm(m) not in have:
                        have.add(norm(m))
                        out.write(m + "\n")
                        added += 1
                out.flush()
                print(f"round {r + 1}/{args.rounds} {topic[:20]}: total new {added}", flush=True)
    print(f"added {added} lines to {EXPANDED}")


def chat_pair(sentence: str, rng: random.Random, opts) -> dict | None:
    pair = gp.make_pair(sentence, rng, opts)
    if pair is None:
        return None
    src = pair["src"]
    if rng.random() < 0.5:
        src = PARTICLE.sub(lambda m: m.group(1), src)  # ireh uu -> irehuu (tgt keeps the space)
    if rng.random() < 0.5:
        src = SUFFIX_HYPHEN.sub(rng.choice([" ", ""]), src)  # 12:30-d -> 12:30 d / 12:30d
    if rng.random() < 0.25:
        src = SPLIT_SUFFIX.sub(lambda m: " " + m.group(1), src)  # huraltai -> hural tai
    # Latin words are copied as typed: a lower-case writer's "pm-d" keeps "pm" in the reference too
    typed = {w.lower(): w for w in LATIN_WORD.findall(src)}
    pair["tgt"] = LATIN_WORD.sub(lambda m: typed.get(m.group().lower(), m.group()), pair["tgt"])
    pair["src"] = src
    return pair


CYR_WORD = re.compile(r"[Ѐ-ӿ]+")


def known_words(min_count: int) -> set[str]:
    """Cyrillic words seen >= min_count times in the v4 synthetic targets (FineWeb-2 / Wikipedia sentences)."""
    from collections import Counter
    counts = Counter()
    with open(ROOT / "data" / "v4" / "train.jsonl", encoding="utf-8-sig") as f:
        for ln in f:
            counts.update(w.lower() for w in CYR_WORD.findall(json.loads(ln)["tgt"]))
    return {w for w, c in counts.items() if c >= min_count}


def cmd_pairs(args):
    gp.CHAT_ABBREV.update(EXTRA_ABBREV)
    gp.CHAT_RE = re.compile(r"(?<![Ѐ-ӿ])(" + "|".join(sorted(map(re.escape, gp.CHAT_ABBREV), key=len, reverse=True))
                            + r")(?![Ѐ-ӿ])", re.IGNORECASE)
    banned = eval_sentences()
    seeds = read_lines(SEEDS) + read_lines(SEEDS2)
    boost = {norm(s) for s in read_lines(SEEDS2)}  # targeted lines: 3x the Latin variants
    vocab = known_words(args.vocab_min) | {w.lower() for s in seeds for w in CYR_WORD.findall(s)}
    lines, seen, dropped = [], set(), []
    for s in seeds + read_lines(EXPANDED):
        k = norm(s)
        if not k or k in seen or k in banned:
            continue
        seen.add(k)
        # Gemma lines: every Cyrillic word must be a known spelling, and no hyphen after a Cyrillic word
        if s not in seeds and (re.search(r"[Ѐ-ӿ]-", s)
                               or any(w.lower() not in vocab for w in CYR_WORD.findall(s))):
            dropped.append(s)
            continue
        lines.append(s)
    (OUT / "dropped_cyr.txt").write_text("\n".join(dropped) + "\n", encoding="utf-8")
    print(f"kept {len(lines)} lines, dropped {len(dropped)} Gemma lines (see {OUT / 'dropped_cyr.txt'})")
    opts = SimpleNamespace(p_mns=0.03, p_chat=1.0, p_stretch=0.1, p_emoji=0.15, p_english=0.05, max_bytes=254)
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "valid": 0}
    with (OUT / "train.jsonl").open("w", encoding="utf-8") as ftr, (OUT / "valid.jsonl").open("w", encoding="utf-8") as fva:
        for s in lines:
            split = "valid" if gp.split_of(norm(s), args.valid_frac, 0.0) == "valid" else "train"
            reps = args.variants * (3 if norm(s) in boost else 1) if split == "train" else 1
            for _ in range(reps):
                p = chat_pair(s, rng, opts)
                if p:
                    (ftr if split == "train" else fva).write(json.dumps(p, ensure_ascii=False) + "\n")
                    counts[split] += 1
    print(f"{len(lines)} Cyrillic lines -> {counts} pairs in {OUT}")


def cmd_test(args):
    rows = []
    for path in sorted(glob.glob(str(ROOT / "data" / "raw" / "dummy_chat_*.json"))):
        chat = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        for m in chat["messages"]:
            tgt = m["text_cyr"]
            if not re.search(r"[.,!?;]", m["text"]):  # writer typed no punctuation -> none in the reference
                tgt = re.sub(r"\s+", " ", re.sub(r"[.,!?;]", " ", tgt)).strip()
            rows.append({"src": m["text"], "tgt": tgt, "source": f"{chat['chat_id']}#{m['id']}"})
    out = ROOT / "data" / "chat_test.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} messages to {out}")


def cmd_mix(args):
    def pairs(path, n=None):
        out = []
        with open(ROOT / path, encoding="utf-8-sig") as f:
            for ln in f:
                if ln.strip():
                    d = json.loads(ln)
                    out.append(json.dumps({"src": d["src"], "tgt": d["tgt"]}, ensure_ascii=False))
                    if n and len(out) >= n:
                        break
        return out

    chat, real, synth = pairs("data/chat/train.jsonl"), pairs("data/real_train.jsonl"), pairs("data/v4/train.jsonl", 10000)
    rows = chat + real * 20 + synth
    random.Random(42).shuffle(rows)
    (ROOT / "data" / "chat_mix.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    dev = pairs("data/real_dev.jsonl") + pairs("data/chat/valid.jsonl")[:300]
    (ROOT / "data" / "chat_dev_mix.jsonl").write_text("\n".join(dev) + "\n", encoding="utf-8")
    print(f"chat_mix: {len(chat)} chat + {len(real)} real x20 + {len(synth)} synthetic; dev: {len(dev)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("expand")
    e.add_argument("--model", default="gemma3:12b")
    e.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    e.add_argument("--rounds", type=int, default=6)
    e.add_argument("--per-call", type=int, default=30)
    e.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("pairs")
    p.add_argument("--variants", type=int, default=6, help="Latin spellings per train line")
    p.add_argument("--valid-frac", type=float, default=0.1)
    p.add_argument("--vocab-min", type=int, default=3, help="min corpus count for a Gemma line's Cyrillic words")
    p.add_argument("--seed", type=int, default=42)
    sub.add_parser("test")
    sub.add_parser("mix")
    args = ap.parse_args(argv)
    {"expand": cmd_expand, "pairs": cmd_pairs, "test": cmd_test, "mix": cmd_mix}[args.cmd](args)


if __name__ == "__main__":
    main()
