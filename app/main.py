"""SRT 字幕时间校准工具后端。

核心逻辑：根据用户给出的若干「原字幕时刻 -> 视频实际时刻」锚点，
按原时刻排序后在相邻锚点间做线性插值，两端按最近一段的斜率外推，
统一换算每条字幕的起止时间，并检测负时间 / 起止倒置 / 与下一条重叠。
"""
import re
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

app = FastAPI(title="SRT 字幕时间校准")

STATIC_DIR = Path(__file__).parent / "static"

TIME_RE = re.compile(r"^\s*(\d{1,3}):([0-5]?\d):([0-5]?\d)[,.](\d{1,3})\s*$")
ARROW_RE = re.compile(r"-->")


# ---------------------------------------------------------------- 时间工具

def parse_time(s: str) -> int:
    """把 'HH:MM:SS,mmm' 解析为毫秒整数，失败抛 ValueError。"""
    m = TIME_RE.match(s or "")
    if not m:
        raise ValueError(f"无法解析时间: {s!r}（应为 HH:MM:SS,mmm）")
    h, mnt, sec, ms = (int(g) for g in m.groups())
    ms = int(str(m.group(4)).ljust(3, "0"))  # 允许 1~3 位毫秒
    return ((h * 60 + mnt) * 60 + sec) * 1000 + ms


def fmt_time(ms: int) -> str:
    """毫秒格式化为 SRT 时间；负数带负号以便预览中识别。"""
    sign = "-" if ms < 0 else ""
    ms = abs(int(round(ms)))
    h, ms = divmod(ms, 3600_000)
    mnt, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{sign}{h:02d}:{mnt:02d}:{sec:02d},{ms:03d}"


# ---------------------------------------------------------------- SRT 解析

def parse_srt(text: str) -> List[dict]:
    """解析 SRT，保留序号与正文。返回 [{index, start, end, lines}]。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n﻿ ")
    if not text:
        raise ValueError("SRT 内容为空")
    entries = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = [ln for ln in block.strip("\n").split("\n") if ln.strip() != ""]
        if not lines:
            continue
        # 找到包含 --> 的时间行，其上一行（若有）为序号
        ti = next((i for i, ln in enumerate(lines) if ARROW_RE.search(ln)), None)
        if ti is None:
            raise ValueError(f"缺少时间行 '-->'：{lines[0][:40]!r}")
        parts = ARROW_RE.split(lines[ti])
        if len(parts) != 2:
            raise ValueError(f"时间行格式错误：{lines[ti]!r}")
        # 去掉时间行中可能附带的定位信息（如 'X1:.. X2:..'）
        start_s = parts[0].strip()
        end_s = parts[1].strip().split()[0] if parts[1].strip() else ""
        try:
            start, end = parse_time(start_s), parse_time(end_s)
        except ValueError as e:
            raise ValueError(f"第 {len(entries) + 1} 块：{e}")
        index = lines[0].strip() if ti > 0 else str(len(entries) + 1)
        entries.append({
            "index": index,
            "start": start,
            "end": end,
            "lines": lines[ti + 1:],
        })
    if not entries:
        raise ValueError("未解析到任何字幕块")
    return entries


# ---------------------------------------------------------------- 锚点与映射

def build_mapper(anchors_ms: List[tuple]):
    """anchors_ms: [(orig_ms, actual_ms)]，已排序且已校验。
    返回分段线性映射函数：相邻锚点线性插值，两端按最近一段斜率外推。"""
    pts = anchors_ms

    def map_t(t: float) -> float:
        if t <= pts[0][0]:
            (o0, a0), (o1, a1) = pts[0], pts[1]
        elif t >= pts[-1][0]:
            (o0, a0), (o1, a1) = pts[-2], pts[-1]
        else:
            for i in range(len(pts) - 1):
                if pts[i][0] <= t <= pts[i + 1][0]:
                    (o0, a0), (o1, a1) = pts[i], pts[i + 1]
                    break
        slope = (a1 - a0) / (o1 - o0)
        return a0 + (t - o0) * slope

    return map_t


def validate_anchors(raw: List["Anchor"]) -> List[tuple]:
    """校验并排序锚点，返回 [(orig_ms, actual_ms)]。失败抛 ValueError。"""
    if len(raw) < 2:
        raise ValueError("至少需要 2 个锚点")
    pts = []
    for i, a in enumerate(raw, 1):
        try:
            pts.append((parse_time(a.orig), parse_time(a.actual)))
        except ValueError as e:
            raise ValueError(f"锚点 {i}：{e}")
    pts.sort(key=lambda p: p[0])
    for i in range(1, len(pts)):
        if pts[i][0] == pts[i - 1][0]:
            raise ValueError(f"锚点原时刻重复：{fmt_time(pts[i][0])}")
        if pts[i][1] <= pts[i - 1][1]:
            raise ValueError(
                f"映射后的实际时刻必须严格递增：{fmt_time(pts[i - 1][1])} 之后不能是 {fmt_time(pts[i][1])}"
            )
    return pts


# ---------------------------------------------------------------- 锚点异常定位

def _lis_len(items: List[tuple]) -> int:
    """items: [(orig_ms, actual_ms)]，已按原时刻排序。
    返回原时刻与实际时刻都严格递增的最长子序列长度。"""
    best = 0
    f = []
    for i, (o, a) in enumerate(items):
        fi = 1
        for j in range(i):
            oj, aj = items[j]
            if oj < o and aj < a:
                fi = max(fi, f[j] + 1)
        f.append(fi)
        best = max(best, fi)
    return best


def _constrained_lis(pts: List[tuple], must_keep: set, must_drop: set) -> int:
    """在「必须保留 must_keep、必须剔除 must_drop」约束下求最长合法子序列长度；
    约束不可行返回 -1。pts: [(orig_ms, actual_ms, line)]，已按原时刻排序。"""
    n = len(pts)
    ks = sorted(must_keep)
    if any(k in must_drop for k in ks):
        return -1
    for a, b in zip(ks, ks[1:]):
        if not (pts[a][0] < pts[b][0] and pts[a][1] < pts[b][1]):
            return -1
    # 必选点把序列切成若干段，段内独立求 LIS（段内点须与两端必选点保持严格递增）
    total = len(ks)
    bounds = [-1] + ks + [n]
    for lo, hi in zip(bounds, bounds[1:]):
        seg = []
        for i in range(lo + 1, hi):
            if i in must_drop:
                continue
            o, a = pts[i][0], pts[i][1]
            if lo != -1 and not (pts[lo][0] < o and pts[lo][1] < a):
                continue
            if hi != n and not (o < pts[hi][0] and a < pts[hi][1]):
                continue
            seg.append((o, a))
        total += _lis_len(seg)
    return total


def suggest_anchor_drops(pts: List[tuple]):
    """从输入锚点中定位需剔除的最少锚点。
    pts: [(orig_ms, actual_ms, line)]，line 为原输入行号（1 起）。
    返回 (drop_lines, kept_pts)：drop_lines 为建议剔除的行号升序列表，
    kept_pts 为保留锚点（按原时刻排序）。最长合法子序列不足 2 点时返回 None。
    决胜规则：子序列最长 → 首尾原时刻锚点保留数量最大 → 剔除行号序列字典序最小。"""
    pts = sorted(pts, key=lambda p: p[0])  # 稳定排序：同原时刻保持输入顺序
    n = len(pts)
    best = _constrained_lis(pts, set(), set())
    if best < 2:
        return None
    first, last = 0, n - 1

    def constrained(mk, md):
        return _constrained_lis(pts, mk, md) == best

    # 首尾原时刻锚点（排序后第一个 / 最后一个）的最大可保留数量
    if constrained({first, last}, set()):
        ends = 2
    elif constrained({first}, {last}) or constrained({last}, {first}):
        ends = 1
    else:
        ends = 0

    def feasible(mk, md):
        if ends == 2:
            return constrained(mk | {first, last}, md)
        if ends == 1:
            return (constrained(mk | {first}, md | {last})
                    or constrained(mk | {last}, md | {first}))
        return constrained(mk, md | {first, last})

    # 按原输入行号升序「能剔则剔」，使剔除行号序列字典序最小
    keep, drop = set(), set()
    for i in sorted(range(n), key=lambda i: pts[i][2]):
        if feasible(keep, drop | {i}):
            drop.add(i)
        else:
            keep.add(i)
    kept = [p for i, p in enumerate(pts) if i not in drop]
    return sorted(pts[i][2] for i in drop), kept


def calibrate(entries: List[dict], map_t) -> List[dict]:
    """换算每条字幕起止时间并打问题标记。"""
    out = []
    for e in entries:
        ns, ne = int(round(map_t(e["start"]))), int(round(map_t(e["end"])))
        flags = []
        if ns < 0 or ne < 0:
            flags.append("负时间")
        if ne <= ns:
            flags.append("起止倒置")
        out.append({**e, "new_start": ns, "new_end": ne, "flags": flags})
    for i in range(len(out) - 1):
        if out[i]["new_end"] > out[i + 1]["new_start"]:
            out[i]["flags"].append("与下一条重叠")
    return out


# ---------------------------------------------------------------- API

class Anchor(BaseModel):
    orig: str    # 原字幕时刻 HH:MM:SS,mmm
    actual: str  # 视频实际时刻 HH:MM:SS,mmm


class CalibrateRequest(BaseModel):
    srt: str
    anchors: List[Anchor]


def _prepare(req: CalibrateRequest):
    """公共预处理，成功返回 (entries, calibrated)，失败抛 400。"""
    try:
        pts = validate_anchors(req.anchors)
        entries = parse_srt(req.srt)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return calibrate(entries, build_mapper(pts))


def _preview_payload(rows: List[dict]) -> dict:
    """由校准结果构造预览响应（/api/preview 与 /api/suggest 共用）。"""
    return {
        "ok": all(not r["flags"] for r in rows),
        "problem_count": sum(1 for r in rows if r["flags"]),
        "entries": [
            {
                "index": r["index"],
                "old": f"{fmt_time(r['start'])} --> {fmt_time(r['end'])}",
                "new": f"{fmt_time(r['new_start'])} --> {fmt_time(r['new_end'])}",
                "flags": r["flags"],
                "text": "\n".join(r["lines"]),
            }
            for r in rows
        ],
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/api/preview")
def preview(req: CalibrateRequest):
    return _preview_payload(_prepare(req))


@app.post("/api/suggest")
def suggest(req: CalibrateRequest):
    """锚点异常定位：校验失败时给出需剔除的最少锚点，并用保留锚点预演校准。"""
    pts = []
    for i, a in enumerate(req.anchors, 1):
        try:
            pts.append((parse_time(a.orig), parse_time(a.actual), i))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"锚点 {i}：{e}")
    # SRT 仅用于预演；解析失败不阻塞定位建议
    srt_error = None
    try:
        entries = parse_srt(req.srt)
    except ValueError as e:
        entries, srt_error = None, str(e)
    try:
        validate_anchors(req.anchors)
        error = None
    except ValueError as e:
        error = str(e)

    result = suggest_anchor_drops(pts)
    if result is None:
        return {
            "possible": False,
            "error": error,
            "message": "按原时刻排序后，原时刻与实际时刻都严格递增的锚点不足 2 个，"
                       "无法预演。请修正或补充锚点。",
        }
    drop_lines, kept = result
    kept_lines = sorted(ln for _, _, ln in kept)
    if error is None:
        message = "锚点校验已通过，无需剔除。"
    else:
        message = (
            f"建议剔除 {len(drop_lines)} 个锚点（第 "
            + "、".join(map(str, drop_lines))
            + " 行），保留第 "
            + "、".join(map(str, kept_lines))
            + f" 行共 {len(kept)} 个锚点即可正常校准。"
        )
    preview_data = None
    if entries is not None:
        preview_data = _preview_payload(
            calibrate(entries, build_mapper([(o, a) for o, a, _ in kept]))
        )
    return {
        "possible": True,
        "error": error,
        "message": message,
        "drop_lines": drop_lines,
        "kept_anchors": [
            {"line": ln, "orig": fmt_time(o), "actual": fmt_time(a)}
            for o, a, ln in sorted(kept, key=lambda p: p[2])
        ],
        "srt_error": srt_error,
        "preview": preview_data,
    }


@app.post("/api/export")
def export(req: CalibrateRequest):
    rows = _prepare(req)
    bad = [r for r in rows if r["flags"]]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"存在 {len(bad)} 条问题字幕（负时间/起止倒置/重叠），已禁止导出",
        )
    blocks = [
        f"{r['index']}\n{fmt_time(r['new_start'])} --> {fmt_time(r['new_end'])}\n"
        + "\n".join(r["lines"])
        for r in rows
    ]
    content = "\n\n".join(blocks) + "\n"
    return Response(
        content=content.encode("utf-8"),
        media_type="application/x-subrip",
        headers={"Content-Disposition": 'attachment; filename="calibrated.srt"'},
    )
