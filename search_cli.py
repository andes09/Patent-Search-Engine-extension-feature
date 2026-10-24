"""search_cli.py — Milestone 5: command-line interface for the patent search engine."""

from __future__ import annotations

import argparse
import sys

from search.engine import SearchEngine, SearchResponse
from search.rerank import TwoPhaseResponse, two_phase_search


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "-", width: int = 72) -> str:
    return char * width


def _print_result(idx: int, r, total: int) -> None:
    print(f"\n{_sep()}")
    print(f"  #{idx + 1} of {total}  | {r.doc_number}  | score {r.score:.6f}")
    print(f"  Title        : {r.title}")
    print(f"  Class        : {r.classification or '(none)'}  | Week: {r.week}")
    print(f"  Found by     : {r.found_by}", end="")
    if r.dense_rank is not None:
        print(f"  (dense rank {r.dense_rank})", end="")
    if r.keyword_rank is not None:
        print(f"  (keyword rank {r.keyword_rank})", end="")
    print()
    if r.matched_terms:
        print(f"  Matched terms: {', '.join(r.matched_terms)}")
    if r.matched_phrases:
        print(f"  Phrases      : {', '.join(r.matched_phrases)}")

    p = r.passage
    label = p.type
    if p.type == "claim":
        label += f" {p.ref}" + (" (independent)" if p.independent else " (dependent)")
    elif p.type == "paragraph":
        label += f" {p.ref}"
    print(f"  Passage [{label}]:")
    # Wrap passage text at ~68 chars for readability
    text = p.text
    line_width = 68
    words = text.split()
    line: list[str] = []
    col = 0
    lines: list[str] = []
    for w in words:
        if col + len(w) + (1 if line else 0) > line_width:
            lines.append(" ".join(line))
            line, col = [w], len(w)
        else:
            line.append(w)
            col += len(w) + (1 if col else 0)
    if line:
        lines.append(" ".join(line))
    for l in lines:
        print(f"    {l}")


def _print_suggestions(suggestions: dict) -> None:
    if not suggestions:
        return
    print(f"\n{_sep('-')}")
    print("  Zero-match suggestions:")
    for term, alts in suggestions.items():
        alts_str = ", ".join(f"{t} ({n})" for t, n in alts)
        print(f"    {term}: 0  →  {alts_str}")


def _print_timings(timings: dict) -> None:
    print(f"\n{_sep('-')}")
    plan = timings.get("plan", "?")
    print(f"  Plan: {plan}")
    keys = ["filter", "query_embedding", "dense", "keyword", "fusion", "total"]
    for key in keys:
        val = timings.get(key)
        if val is not None:
            print(f"  {key:<18}: {val * 1000:7.1f} ms")


def _print_rerank_result(idx: int, r, total: int) -> None:
    """Print a RerankResult with movement annotation."""
    if r.moved > 0:
        move_str = f"+{r.moved}"
    elif r.moved < 0:
        move_str = f"{r.moved}"   # already negative, e.g. -3
    else:
        move_str = "="
    print(f"\n{_sep()}")
    print(
        f"  #{r.final_rank} of {total}  "
        f"(was #{r.stage1_rank}, {move_str})  "
        f"| {r.doc_number}  | rerank {r.rerank_score:.4f}"
    )
    print(f"  Title        : {r.title}")
    print(f"  Class        : {r.classification or '(none)'}  | Week: {r.week}")
    print(f"  Found by     : {r.found_by}", end="")
    if r.dense_rank is not None:
        print(f"  (dense rank {r.dense_rank})", end="")
    if r.keyword_rank is not None:
        print(f"  (keyword rank {r.keyword_rank})", end="")
    print()
    if r.matched_terms:
        print(f"  Matched terms: {', '.join(r.matched_terms)}")
    if r.matched_phrases:
        print(f"  Phrases      : {', '.join(r.matched_phrases)}")

    p = r.passage
    label = p.type
    if p.type == "claim":
        label += f" {p.ref}" + (" (independent)" if p.independent else " (dependent)")
    elif p.type == "paragraph":
        label += f" {p.ref}"
    print(f"  Passage [{label}]:")
    text = p.text
    line_width = 68
    words = text.split()
    line: list[str] = []
    col = 0
    lines: list[str] = []
    for w in words:
        if col + len(w) + (1 if line else 0) > line_width:
            lines.append(" ".join(line))
            line, col = [w], len(w)
        else:
            line.append(w)
            col += len(w) + (1 if col else 0)
    if line:
        lines.append(" ".join(line))
    for ln in lines:
        print(f"    {ln}")


def _print_rerank_timings(timings: dict) -> None:
    print(f"\n{_sep('-')}")
    print(f"  Timing breakdown:")
    print(f"  {'stage1':<18}: {timings.get('stage1_ms', 0):7.1f} ms")
    print(f"  {'rerank':<18}: {timings.get('rerank_ms', 0):7.1f} ms")
    print(f"  {'n_pairs':<18}: {timings.get('n_pairs', 0)}")
    print(f"  {'total':<18}: {timings.get('total_ms', 0):7.1f} ms")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="search_cli.py",
        description="Hybrid patent search engine (ThinkStruct take-home, Part 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python search_cli.py "tire tread groove that reduces road noise" -k 5
  python search_cli.py "wheel rim" --class B60B -k 10
  python search_cli.py --patent 20240001111 -k 5
  python search_cli.py "rubber compound" --keywords "bead seat" sealant -k 5
  python search_cli.py "wheel" --title-has wheel --abstract-has steel --mode hybrid
""",
    )

    p.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Natural-language or claim query text",
    )
    p.add_argument(
        "--patent",
        metavar="DOC_NUMBER",
        default=None,
        help="Use a patent from the dataset as the query seed (by doc_number)",
    )
    p.add_argument(
        "--mode",
        choices=["hybrid", "dense", "keyword"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    p.add_argument(
        "--class",
        dest="class_prefix",
        metavar="PREFIX",
        default=None,
        help="Hard filter: classification must start with PREFIX (e.g. B60C)",
    )
    p.add_argument(
        "--title-has",
        nargs="+",
        metavar="WORD",
        default=None,
        help="Hard filter: every WORD must appear in the title",
    )
    p.add_argument(
        "--abstract-has",
        nargs="+",
        metavar="WORD",
        default=None,
        help="Hard filter: every WORD must appear in the abstract",
    )
    p.add_argument(
        "--title",
        dest="title_exact",
        metavar="TITLE",
        default=None,
        help="Hard filter: exact title match (normalised lowercase)",
    )
    p.add_argument(
        "--keywords",
        nargs="+",
        metavar="KW",
        default=None,
        help='Soft keywords (2× weight); wrap phrases in double quotes: "bead core"',
    )
    p.add_argument(
        "-k",
        type=int,
        default=10,
        metavar="K",
        help="Number of results to return (default: 10)",
    )
    p.add_argument(
        "--plan",
        choices=["auto", "post"],
        default="auto",
        help="Query plan: auto (fast) or post (full-scan baseline)",
    )
    p.add_argument(
        "--data",
        default="data/data/patent_data_small",
        metavar="DIR",
        help="Path to patent JSON directory (default: data/data/patent_data_small)",
    )

    # ---- Two-phase reranking -----------------------------------------------
    p.add_argument(
        "--rerank",
        action="store_true",
        default=False,
        help="Enable stage-2 cross-encoder reranking",
    )
    p.add_argument(
        "--rerank-model",
        dest="rerank_model",
        choices=["minilm", "bge"],
        default="minilm",
        help="Reranker model: minilm (fast) or bge (stronger, slower) [default: minilm]",
    )
    p.add_argument(
        "--rerank-n",
        dest="rerank_n",
        type=int,
        default=100,
        metavar="N",
        help="Number of stage-1 candidates to pass to the reranker [default: 100]",
    )
    p.add_argument(
        "--rerank-mode",
        dest="rerank_mode",
        choices=["pure", "blend"],
        default="pure",
        help="Reranking mode: pure (reranker score only) or blend (RRF) [default: pure]",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args   = parser.parse_args(argv)

    if args.query is None and args.patent is None:
        parser.error("Provide a query string or --patent DOC_NUMBER")

    print(f"Loading index from {args.data!r} …")
    engine = SearchEngine(args.data)
    print("Index ready.\n")

    # Shared filter kwargs passed through to stage 1 (and to two_phase_search).
    filter_kwargs = dict(
        class_prefix=args.class_prefix,
        title_has=args.title_has,
        abstract_has=args.abstract_has,
        title_exact=args.title_exact,
        keywords=args.keywords,
        plan=args.plan,
    )

    try:
        if args.rerank:
            rresp: TwoPhaseResponse = two_phase_search(
                engine,
                query=args.query,
                patent_id=args.patent,
                n_candidates=args.rerank_n,
                k=args.k,
                model=args.rerank_model,
                mode=args.rerank_mode,
                **filter_kwargs,
            )
        else:
            resp: SearchResponse = engine.search(
                query=args.query,
                patent_id=args.patent,
                mode=args.mode,
                k=args.k,
                **filter_kwargs,
            )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    # ---- Header -------------------------------------------------------------
    print(f"{_sep('=')}")
    if args.patent:
        print(f"  Query seed : patent {args.patent}")
    else:
        q_display = args.query if len(args.query) <= 80 else args.query[:77] + "…"
        print(f"  Query      : {q_display}")

    if args.rerank:
        print(
            f"  Rerank     : {args.rerank_model} | mode={args.rerank_mode}"
            f" | n_candidates={args.rerank_n} | k={args.k}"
        )
    else:
        print(f"  Mode       : {args.mode}  | Plan: {args.plan}  | k={args.k}")

    filters = []
    if args.class_prefix:
        filters.append(f"class={args.class_prefix}")
    if args.title_has:
        filters.append(f"title-has={args.title_has}")
    if args.abstract_has:
        filters.append(f"abstract-has={args.abstract_has}")
    if args.title_exact:
        filters.append(f"title-exact={args.title_exact!r}")
    if filters:
        print(f"  Filters    : {', '.join(filters)}")
    if args.keywords:
        print(f"  Keywords   : {args.keywords}")

    if args.rerank:
        print(f"  Results    : {len(rresp.results)}")
        print(f"{_sep('=')}")
        if not rresp.results:
            print("\n  (no results — try broadening filters or the query)")
        else:
            for r in rresp.results:
                _print_rerank_result(r.final_rank - 1, r, len(rresp.results))
        _print_suggestions(rresp.suggestions)
        _print_rerank_timings(rresp.timings)
    else:
        print(f"  Results    : {len(resp.results)}")
        print(f"{_sep('=')}")
        if not resp.results:
            print("\n  (no results — try broadening filters or the query)")
        else:
            for i, r in enumerate(resp.results):
                _print_result(i, r, len(resp.results))
        _print_suggestions(resp.suggestions)
        _print_timings(resp.timings)

    print(f"\n{_sep('=')}")


if __name__ == "__main__":
    main()
