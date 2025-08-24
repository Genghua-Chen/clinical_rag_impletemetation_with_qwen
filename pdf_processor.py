import re
from typing import Optional, Set

def robust_split_runons(
    text: str,
    *,
    max_token_len: int = 6,
    custom_vocab: Optional[Set[str]] = None,
    debug: bool = False
):
    """
    Stronger splitter:
    - Heavy penalty for unknown/long words so 'influenza vaccine field has' wins.
    - Second pass: fallback on any leftover long unknown parts.
    """
    pieces = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
    vocab = {w.lower() for w in (custom_vocab or set())}

    def wordfreq_zipf(w: str):
        try:
            from wordfreq import zipf_frequency
            return zipf_frequency(w, "en")
        except Exception:
            return 0.0  # treat as very-rare if wordfreq missing

    def fallback_split(tok: str):
        try:
            import wordninja
            return " ".join(wordninja.split(tok))
        except Exception:
            try:
                from wordsegment import load, segment
                load()
                return " ".join(segment(tok))
            except Exception:
                return tok

    def best_split_wordfreq(tok: str):
        s = tok.lower()
        n = len(s)
        best = [0.0] + [float("inf")] * n
        back = [-1] * (n + 1)

        def cost(w: str):
            z = wordfreq_zipf(w)
            # base cost: rarer words cost more
            base = 7.0 - min(z, 7.0)
            # strong unknown penalty (if z < 3.2 it’s likely not a standalone English word)
            unknown_pen = 0.0 if z >= 3.2 else (3.0 + (3.2 - z))
            # length penalty so giant chunks don't dominate
            len_pen = max(0, len(w) - 5) * 0.12
            # domain vocab bonus
            dom_bonus = -2.0 if w in vocab else 0.0
            return max(0.001, base + unknown_pen + len_pen + dom_bonus)

        for i in range(1, n + 1):
            for j in range(max(0, i - 24), i):  # consider up to 24-char words
                w = s[j:i]
                if not w.isalpha() or len(w) < 2:
                    continue
                c = best[j] + cost(w)
                if c < best[i]:
                    best[i] = c
                    back[i] = j

        if back[-1] == -1:
            return None

        parts = []
        k = n
        while k > 0:
            j = back[k]
            parts.append(s[j:k])
            k = j
        parts.reverse()
        if tok[0].isupper():
            parts[0] = parts[0].capitalize()
        return parts

    out = []
    for p in pieces:
        if p.isalpha() and len(p) > max_token_len and p.lower() not in vocab:
            parts = best_split_wordfreq(p)
            if parts:
                # second pass: fix any leftover long/unknown segments with fallback
                refined = []
                for seg in parts:
                    z = wordfreq_zipf(seg)
                    if len(seg) > max_token_len and z < 3.2:
                        fallback = fallback_split(seg)
                        refined.extend(fallback.split(" "))
                        if debug and fallback != seg:
                            print(f"[fallback] {seg} -> {fallback}")
                    else:
                        refined.append(seg)
                if debug and " ".join(refined) != p:
                    print(f"[split] {p} -> {' '.join(refined)}")
                # re-apply leading cap if needed
                if p[0].isupper() and refined:
                    refined[0] = refined[0].capitalize()
                out.append(" ".join(refined))
            else:
                out.append(fallback_split(p))
        else:
            out.append(p)
    return "".join(out)


s = "andflexibilityofmanufacturing."
print(robust_split_runons(
    s,
    max_token_len=3,
    custom_vocab={"BEVS"},
    debug=False
))
