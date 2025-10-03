import os
import time
import math
import asyncio
# workflows/query_workflow.py
# Workflow orchestrating the whole query to answer pipeline

from .base import Workflow
from events import (
    StartQueryEvent,
    QueryPlanningCompleteEvent,
    StopEvent,
    SearchCompleteEvent,
    RerankCompleteEvent,
    WritingCompleteEvent,
    RewriteEvent
)
from agents.query_planner_agent import QueryPlannerAgent
from config import Config
from agents.searcher_agent import SearcherAgent
from agents.reranker_agent import RerankerAgent
from agents.writer_agent import WriterAgent
from agents.verifier_agent import VerifierAgent
from agents.direct_answer_agent import DirectAnswerAgent


class QueryWorkflow(Workflow):
    """Orchestrates the real-time, interactive query process."""

    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.query_planner_agent = QueryPlannerAgent(config)
        self.searcher_agent = SearcherAgent(config)
        self.reranker_agent = RerankerAgent(config)
        self.writer_agent = WriterAgent(config)
        self.verifier_agent = VerifierAgent(config)
        self.direct_answer_agent = DirectAnswerAgent(config)

        self.add_listener(StartQueryEvent, self.start_query_planning)
        self.add_listener(QueryPlanningCompleteEvent, self.start_search)
        self.add_listener(SearchCompleteEvent, self.start_reranking)
        self.add_listener(RerankCompleteEvent, self.handle_writing_request)
        self.add_listener(RewriteEvent, self.handle_writing_request)
        self.add_listener(WritingCompleteEvent, self.start_verification)

    def _set_status(self, phase: str) -> None:
        callback = self.context.get('set_status')
        if callable(callback):
            try:
                callback(phase)
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[Workflow Status] Failed to update status: {exc}")

    def _build_result(self, final_answer: str, verification_feedback: str | None = None) -> dict:
        """Construct the StopEvent result payload with common metadata, including
        retrieval confidence signals for evaluation and 'used_retrieval' flag.
        """
        return {
            "final_answer": final_answer,
            "verification_feedback": verification_feedback,
            "timings": self.context.get('timings', {}),
            "sources": self.context.get('source_refs', []),
            "used_retrieval": bool(self.context.get('source_refs')),
            "retrieval_confidence": self.context.get('retrieval_confidence_label', 'none'),
            "retrieval_confidence_features": self.context.get('retrieval_confidence_features', {}),
            "requires_retrieval": self.context.get('requires_retrieval', None),
        }

    def collect_sources(self, nodes):
        sources = []
        seen = set()
        import re
        query_text = (self.context.get('last_query') or '').strip()
        stopwords = {
            'the','a','an','and','or','but','if','then','else','for','to','of','in','on','at','by','with',
            'is','are','was','were','be','been','being','this','that','these','those','it','as','from','about',
            'what','which','who','whom','how','why','when','where','can','could','should','would','do','does','did'
        }
        query_tokens = []
        if query_text:
            query_tokens = [t for t in re.split(r"\W+", query_text) if t and t.lower() not in stopwords]
            query_tokens.sort(key=lambda s: (-len(s), s.lower()))

        def choose_search_term(raw_text: str) -> str | None:
            if not raw_text:
                return None
            lower_text = raw_text.lower()
            for qt in query_tokens:
                if qt.lower() in lower_text and len(qt) >= 4:
                    return qt
            m = re.search(r"[A-Za-z][A-Za-z0-9\.-]{4,}", raw_text)
            if m:
                return m.group(0)
            words = [w for w in re.split(r"\s+", raw_text.strip()) if w]
            if words:
                return " ".join(words[:3])[:80]
            return None

        def build_query_aligned_snippet(text: str, tokens: list[str], fallback_keyword: str | None, window: int = 1200) -> str | None:
            """Return a window of text centered on up to two strong query token matches.
            - Prefers longest tokens first.
            - If two matches are far apart, stitch two smaller spans with an ellipsis.
            - Falls back to keyword or head of text.
            Bounds near sentence edges for readability when possible.
            """
            if not text:
                return None
            lower = text.lower()
            matches: list[int] = []
            seen_terms: set[str] = set()
            for qt in tokens or []:
                term = qt.lower()
                if term in seen_terms:
                    continue
                idx = lower.find(term)
                if idx != -1:
                    matches.append(idx)
                    seen_terms.add(term)
                if len(matches) >= 2:
                    break

            # Fallback: keyword
            if not matches and fallback_keyword:
                i = lower.find(str(fallback_keyword).lower())
                if i != -1:
                    matches.append(i)

            # If still no match, return head of text
            if not matches:
                return text[:window]

            def _window_around(idx: int, span: int) -> tuple[int, int]:
                half = max(200, span // 2)
                s = max(0, idx - half)
                e = min(len(text), idx + half)
                # Snap to sentence boundaries if possible
                lb = max(text.rfind('. ', 0, s), text.rfind('\n', 0, s))
                if lb != -1:
                    s = max(0, lb + 1)
                rb_dot = text.find('. ', e)
                rb_nl = text.find('\n', e)
                cands = [c for c in [rb_dot, rb_nl] if c != -1]
                if cands:
                    e = min(min(cands) + 1, len(text))
                return s, e

            if len(matches) == 1:
                s, e = _window_around(matches[0], window)
                return (text[s:e] or text[:window]).strip()

            # Two matches: if close, take one combined window; else, stitch two spans
            i1, i2 = sorted(matches[:2])
            if i2 - i1 < int(window * 0.8):
                s, e = _window_around((i1 + i2) // 2, window)
                return (text[s:e] or text[:window]).strip()
            else:
                s1, e1 = _window_around(i1, max(400, window // 3))
                s2, e2 = _window_around(i2, max(400, window // 3))
                part1 = text[s1:e1].strip()
                part2 = text[s2:e2].strip()
                stitched = (part1 + " ... " + part2).strip()
                return stitched or text[:window]
        for idx, node_with_score in enumerate(nodes or [], start=1):
            node = node_with_score.node
            metadata = getattr(node, "metadata", {}) or {}
            source_path = metadata.get("source_path") or metadata.get("file_path")
            source_file = metadata.get("source_file") or (
                os.path.basename(source_path) if source_path else None
            )
            if not source_file:
                continue

            raw_page_number = metadata.get("source_page_number")
            page_label = metadata.get("source_page_label") or metadata.get("page_label")
            # Compute a reliable 1-based page number for PDF viewers
            page_number = None
            if raw_page_number is not None:
                try:
                    pn = int(raw_page_number)
                    page_number = pn + 1 if pn <= 0 else pn
                except Exception:
                    page_number = None
            if page_number is None and page_label is not None:
                try:
                    if str(page_label).isdigit():
                        page_number = int(page_label)
                except Exception:
                    page_number = None
            raw_content = node.get_content()
            excerpt = metadata.get("source_excerpt") or (raw_content.strip()[:200] if raw_content else None)
            highlight_id = metadata.get("source_node_id") or getattr(node, "id_", None)
            highlight_keyword = metadata.get("highlight_keyword") or choose_search_term(raw_content)
            # Build a query-aligned highlight window used by the evaluator and UI
            highlight_text = metadata.get("highlight_text")
            aligned = build_query_aligned_snippet(raw_content or '', query_tokens, highlight_keyword)
            if aligned and len(aligned) >= 40:
                highlight_text = aligned
            elif not highlight_text:
                highlight_text = excerpt

            key = (source_file, page_number, highlight_id)
            if key in seen:
                continue
            seen.add(key)

            sources.append({
                "id": highlight_id,
                "file": source_file,
                "path": source_path,
                "page": page_number,
                "page_label": page_label,
                "score": float(node_with_score.score) if node_with_score.score is not None else None,
                "excerpt": excerpt,
                "highlight_text": highlight_text,
                "highlight_keyword": highlight_keyword,
                "rank": idx,
            })
        return sources

    def _parse_citations(self, answer: str):
        """Extract citations of the form [file.pdf, p.X] or [file.pdf]. Returns a list of (file, page_str|None)."""
        import re
        cites = []
        if not answer:
            return cites
        for m in re.finditer(r"\[([^\]]+)\]", answer):
            inner = m.group(1).strip()
            # Split on ';' to allow [a][b] style to be separate by regex; here we handle only inner content
            parts = [p.strip() for p in inner.split(',')]
            if not parts:
                continue
            file = parts[0]
            page = None
            if len(parts) > 1:
                tail = ','.join(parts[1:])
                pm = re.search(r"(p\.?\s*|page\s*)([0-9]+)", tail, flags=re.IGNORECASE)
                if pm:
                    page = pm.group(2)
            if file:
                cites.append((file, page))
        return cites

    def _page_biased_sources(self, answer: str, nodes):
        """Build a source list biased to (file,page) pairs cited in the answer,
        with sentence-aligned highlight windows so the evaluator sees the exact evidence.
        Falls back to collect_sources if nothing is parsed/matched.
        """
        import re

        # --- Helper: extract (sentence -> citations[]) from answer ---
        def _extract_cited_sentences(text: str):
            if not text:
                return []
            # Split answer into sentences conservatively
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s and not s.isspace()]
            result = []
            for s in sents:
                cites = []
                for m in re.finditer(r"\[([^\]]+)\]", s):
                    inner = m.group(1).strip()
                    parts = [p.strip() for p in inner.split(',')]
                    if not parts:
                        continue
                    file = parts[0]
                    page = None
                    if len(parts) > 1:
                        tail = ','.join(parts[1:])
                        pm = re.search(r"(p\.?\s*|page\s*)([0-9]+)", tail, flags=re.IGNORECASE)
                        if pm:
                            page = pm.group(2)
                    if file:
                        cites.append((file, page))
                if cites:
                    result.append((s, cites))
            return result

        cited = _extract_cited_sentences(answer)
        if not cited:
            # Fallback to generic sources if no parsable citations
            return self.collect_sources(nodes)

        # --- Build quick index of nodes by (file, page_label) ---
        indexed = {}
        for nws in nodes or []:
            node = nws.node
            md = getattr(node, 'metadata', {}) or {}
            f = md.get('source_file') or md.get('file_name')
            pl = md.get('source_page_label') or md.get('page_label')
            key = (f, str(pl) if pl is not None else None)
            indexed.setdefault(key, []).append(nws)

        # --- Lazy load full node index from disk for fallback when page not in top-K ---
        # Maps: (file, page_label) -> list[nodes], and file -> list[nodes]
        full_index = self.context.get('_all_nodes_index')
        full_file_map = self.context.get('_all_nodes_by_file')
        if full_index is None or full_file_map is None:
            try:
                import pickle
                from config import Config as _Cfg
                with open(_Cfg.NODES_PATH, 'rb') as _h:
                    all_nodes = pickle.load(_h)
                idx = {}
                by_file = {}
                for n in all_nodes or []:
                    md = getattr(n, 'metadata', {}) or {}
                    f = md.get('source_file') or md.get('file_name')
                    pl = md.get('source_page_label') or md.get('page_label')
                    k = (f, str(pl) if pl is not None else None)
                    if f:
                        by_file.setdefault(f, []).append(n)
                    idx.setdefault(k, []).append(n)
                full_index = idx
                full_file_map = by_file
                self.context['_all_nodes_index'] = full_index
                self.context['_all_nodes_by_file'] = full_file_map
            except Exception:
                full_index = {}
                full_file_map = {}

        # Roman numeral to int (basic) for tolerant matching
        def _roman_to_int(s: str) -> int | None:
            if not isinstance(s, str):
                return None
            s = s.upper()
            vals = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
            try:
                total = 0
                prev = 0
                for ch in reversed([c for c in s if c in vals]):
                    v = vals[ch]
                    if v < prev:
                        total -= v
                    else:
                        total += v
                        prev = v
                return total if total > 0 else None
            except Exception:
                return None

        def _choose_node(file: str, page: str | None, sentence: str):
            # Prefer exact (file,page) match, else (file,None), else nearest page by tolerance
            key_exact = (file, str(page) if page is not None else None)
            candidates = indexed.get(key_exact) or []
            # Try off-by-one numeric pages and roman numerals
            if not candidates and page is not None:
                try:
                    p = int(page)
                    for delta in (-1, 1):
                        key_alt = (file, str(p + delta))
                        if key_alt in indexed:
                            candidates = indexed[key_alt]
                            break
                except Exception:
                    # Try roman → int → string
                    p2 = _roman_to_int(page)
                    if p2 is not None:
                        key_alt = (file, str(p2))
                        candidates = indexed.get(key_alt) or []
            # Validate candidate content against sentence tokens; if weak, fall back to BM25 within file
            def _sentence_tokens(raw: str):
                import re as _re
                split_tokens = [t for t in _re.split(r"\W+", (raw or '').lower()) if t]
                upper_tokens = [t for t in _re.findall(r"\b[A-Z]{2,}\b", raw or '')]
                numeric_tokens = [m.group(0) for m in _re.finditer(r"\b\d+(?:\.\d+)?\b", raw or '')]
                toks = [t for t in split_tokens if len(t) >= 4]
                toks.extend([t.lower() for t in upper_tokens])
                toks.extend([t for t in numeric_tokens])
                return set(toks)

            def _has_overlap(nws_obj, sent: str, min_overlap: int) -> bool:
                try:
                    text = nws_obj.node.get_content() or ''
                except Exception:
                    text = ''
                import re as _re
                doc_tokens = set([t for t in _re.split(r"\W+", text.lower()) if t])
                s_tokens = _sentence_tokens(sent)
                return len(doc_tokens & s_tokens) >= min_overlap

            if candidates:
                MIN_OVERLAP = getattr(self.config, 'CITATION_BM25_MIN_OVERLAP', 2)
                top = sorted(candidates, key=lambda x: (x.score is None, -(x.score or 0)))[0]
                if _has_overlap(top, sentence, MIN_OVERLAP):
                    return top
                # Try other candidates by score
                for alt in sorted(candidates, key=lambda x: (x.score is None, -(x.score or 0)))[1:3]:
                    if _has_overlap(alt, sentence, MIN_OVERLAP):
                        return alt

            if not candidates:
                # Aggregate all nodes for same file (any page)
                agg = []
                for (f, pl), lst in indexed.items():
                    if f == file:
                        agg.extend(lst)
                candidates = agg
            if not candidates:
                # Fallback to full corpus nodes for the same (file, page) or same file using sentence match
                # Build a lightweight BM25 over nodes of the file to choose best span
                try:
                    from rank_bm25 import BM25Okapi as _BM25
                except Exception:
                    class _BM25:  # minimal fallback
                        def __init__(self, corpus):
                            from collections import Counter
                            import math
                            self.corpus = corpus
                            self.N = len(corpus)
                            self.df = Counter()
                            for doc in corpus:
                                for term in set(doc):
                                    self.df[term] += 1
                            self.avgdl = (sum(len(doc) for doc in corpus) / self.N) if self.N else 1.0
                            self.k1 = 1.5
                            self.b = 0.75
                        def _idf(self, term):
                            import math
                            n = self.df.get(term, 0)
                            return math.log(1 + (self.N - n + 0.5) / (n + 0.5)) if self.N else 0.0
                        def get_scores(self, query):
                            scores = [0.0]*self.N
                            for i, doc in enumerate(self.corpus):
                                dl = len(doc) or 1
                                tf = {}
                                for t in doc:
                                    tf[t] = tf.get(t,0)+1
                                s = 0.0
                                for q in query or []:
                                    f = tf.get(q,0)
                                    if not f: continue
                                    idf = self._idf(q)
                                    denom = f + self.k1*(1 - self.b + self.b*dl/self.avgdl)
                                    s += idf * (f*(self.k1+1))/denom
                                scores[i]=s
                            return scores
                import re as _re
                file_nodes = full_file_map.get(file) or []
                if not file_nodes:
                    return None
                # Tokenize sentence with relaxed policy (match highlighter)
                raw_sentence = sentence or ''
                split_tokens = [t for t in _re.split(r"\W+", raw_sentence.lower()) if t]
                upper_tokens = [t for t in _re.findall(r"\b[A-Z]{2,}\b", raw_sentence)]
                numeric_tokens = [m.group(0) for m in _re.finditer(r"\b\d+(?:\.\d+)?\b", raw_sentence)]
                q_tokens = [t for t in split_tokens if len(t) >= 4]
                q_tokens.extend([t.lower() for t in upper_tokens])
                q_tokens.extend([t for t in numeric_tokens])
                # Build corpus tokens
                corpus_tokens = []
                for n in file_nodes:
                    txt = (getattr(n, 'get_content', lambda: '')() or '')
                    corpus_tokens.append([t for t in _re.split(r"\W+", txt.lower()) if t])
                bm = _BM25(corpus_tokens) if corpus_tokens else None
                if bm:
                    scores = bm.get_scores(q_tokens)
                    if scores:
                        best_idx = max(range(len(scores)), key=lambda i: scores[i])
                        best_node = file_nodes[best_idx]
                        # Wrap in a small object with .node/.score
                        class _NWS:
                            def __init__(self, node, score=None):
                                self.node = node
                                self.score = score
                        return _NWS(best_node, scores[best_idx])
                return None
            # As last resort, return best by reranker score
            return sorted(candidates, key=lambda x: (x.score is None, -(x.score or 0)))[0]

        # --- Build sentence-aligned snippet around sentence tokens ---
        def _build_sentence_aligned_snippet(text: str, sentence: str, window: int = 1400) -> str:
            if not text:
                return ''
            lower = text.lower()
            import re as _re
            raw_sentence = (sentence or '')
            # Token policy: include long tokens (>=4), ALL-CAPS acronyms (>=2), and numeric tokens (e.g., 15, 2.2)
            split_tokens = [t for t in _re.split(r"\W+", raw_sentence.lower()) if t]
            upper_tokens = [t for t in _re.findall(r"\b[A-Z]{2,}\b", raw_sentence)]
            numeric_tokens = [m.group(0) for m in _re.finditer(r"\b\d+(?:\.\d+)?\b", raw_sentence)]
            tokens = [t for t in split_tokens if len(t) >= 4]
            tokens.extend([t.lower() for t in upper_tokens])
            tokens.extend([t for t in numeric_tokens])
            # Find up to two match positions for strongest tokens (longest first)
            tokens.sort(key=lambda s: (-len(s), s))
            matches = []
            seen = set()
            for t in tokens:
                if t in seen:
                    continue
                i = lower.find(t)
                if i != -1:
                    matches.append(i)
                    seen.add(t)
                if len(matches) >= 2:
                    break
            # Fallback: head of text
            if not matches:
                return text[:window]
            def _bounds(idx: int, span: int) -> tuple[int,int]:
                half = max(200, span // 2)
                s = max(0, idx - half)
                e = min(len(text), idx + half)
                lb = max(text.rfind('. ', 0, s), text.rfind('\n', 0, s))
                if lb != -1:
                    s = max(0, lb + 1)
                rb_dot = text.find('. ', e)
                rb_nl = text.find('\n', e)
                cands = [c for c in [rb_dot, rb_nl] if c != -1]
                if cands:
                    e = min(min(cands) + 1, len(text))
                return s, e
            if len(matches) == 1:
                s, e = _bounds(matches[0], window)
                return (text[s:e] or text[:window]).strip()
            i1, i2 = sorted(matches[:2])
            if i2 - i1 < int(window * 0.8):
                s, e = _bounds((i1 + i2)//2, window)
                return (text[s:e] or text[:window]).strip()
            # stitch two spans
            s1, e1 = _bounds(i1, max(400, window//3))
            s2, e2 = _bounds(i2, max(400, window//3))
            part1 = text[s1:e1].strip()
            part2 = text[s2:e2].strip()
            stitched = (part1 + ' ... ' + part2).strip()
            return stitched or text[:window]

        sources = []
        seen = set()
        for sent, cites in cited:
            for file, page in cites:
                chosen = _choose_node(file, page, sent)
                if not chosen:
                    continue
                node = chosen.node
                md = getattr(node, 'metadata', {}) or {}
                source_path = md.get('source_path') or md.get('file_path')
                source_file = md.get('source_file') or md.get('file_name') or file
                page_label = md.get('source_page_label') or md.get('page_label')
                raw_page_number = md.get('source_page_number')
                page_number = None
                if raw_page_number is not None:
                    try:
                        pn = int(raw_page_number)
                        page_number = pn + 1 if pn <= 0 else pn
                    except Exception:
                        page_number = None
                key = (source_file, str(page_label) if page_label is not None else None, getattr(node, 'id_', None))
                if key in seen:
                    continue
                seen.add(key)
                node_text = node.get_content() or ''
                highlight_text = _build_sentence_aligned_snippet(node_text, sent, window=1400)
                sources.append({
                    'id': md.get('source_node_id') or getattr(node, 'id_', None),
                    'file': source_file,
                    'path': source_path,
                    'page': page_number,
                    'page_label': page_label,
                    'score': float(chosen.score) if chosen.score is not None else None,
                    'excerpt': md.get('source_excerpt'),
                    'highlight_text': highlight_text,
                    'highlight_keyword': md.get('highlight_keyword'),
                    'rank': len(sources) + 1,
                })
        # Light support for uncited sentences: attach snippets for a small number of uncited claims
        try:
            import re as _re
            all_sents = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", (answer or '').strip()) if s and not s.isspace()]
            cited_set = set(s for s, _ in cited)
            limit_uncited = getattr(self.config, 'UNCITED_SENTENCE_SUPPORT', 2)
            uncited_sents = [s for s in all_sents if s not in cited_set][:max(0, limit_uncited)]
            if uncited_sents:
                # Build BM25 over reranked nodes (or fallback to full corpus by file selection later)
                try:
                    from rank_bm25 import BM25Okapi as _BM25U
                except Exception:
                    class _BM25U:
                        def __init__(self, corpus):
                            from collections import Counter
                            self.corpus = corpus; self.N=len(corpus); self.df=Counter(); self.k1=1.5; self.b=0.75; self.avgdl=(sum(len(d) for d in corpus)/self.N) if self.N else 1
                            for d in corpus:
                                for t in set(d): self.df[t]+=1
                        def _idf(self,t):
                            import math; n=self.df.get(t,0); return math.log(1+(self.N-n+0.5)/(n+0.5)) if self.N else 0.0
                        def get_scores(self,q):
                            scores=[0.0]*self.N
                            for i,doc in enumerate(self.corpus):
                                dl=len(doc) or 1; tf={}
                                for t in doc: tf[t]=tf.get(t,0)+1
                                s=0.0
                                for qt in q:
                                    f=tf.get(qt,0);
                                    if not f: continue
                                    idf=self._idf(qt)
                                    denom=f + self.k1*(1 - self.b + self.b*dl/self.avgdl)
                                    s+= idf*(f*(self.k1+1))/denom
                                scores[i]=s
                            return scores
                # Tokenizer
                def _tok(s): return [t for t in _re.split(r"\W+", (s or '').lower()) if t]
                # Build corpus
                corpus_nodes = [nws.node for nws in (nodes or [])]
                corpus_tokens = [_tok(getattr(n, 'get_content', lambda: '')() or '') for n in corpus_nodes]
                bm = _BM25U(corpus_tokens) if corpus_tokens else None
                for us in uncited_sents:
                    if not bm:
                        break
                    q_tokens = _tok(us)
                    if not q_tokens:
                        continue
                    scores = bm.get_scores(q_tokens)
                    if not scores:
                        continue
                    best_i = max(range(len(scores)), key=lambda i: scores[i])
                    best_node = corpus_nodes[best_i]
                    md = getattr(best_node, 'metadata', {}) or {}
                    source_file = md.get('source_file') or md.get('file_name')
                    source_path = md.get('source_path') or md.get('file_path')
                    page_label = md.get('source_page_label') or md.get('page_label')
                    raw_page_number = md.get('source_page_number')
                    page_number = None
                    if raw_page_number is not None:
                        try:
                            pn = int(raw_page_number)
                            page_number = pn + 1 if pn <= 0 else pn
                        except Exception:
                            page_number = None
                    key = (source_file, str(page_label) if page_label is not None else None, getattr(best_node, 'id_', None))
                    if key in seen:
                        continue
                    highlight_text = _build_sentence_aligned_snippet(getattr(best_node,'get_content',lambda:'' )() or '', us, window=1400)
                    sources.append({
                        'id': md.get('source_node_id') or getattr(best_node, 'id_', None),
                        'file': source_file,
                        'path': source_path,
                        'page': page_number,
                        'page_label': page_label,
                        'score': None,
                        'excerpt': md.get('source_excerpt'),
                        'highlight_text': highlight_text,
                        'highlight_keyword': md.get('highlight_keyword'),
                        'rank': len(sources) + 1,
                    })
                    seen.add(key)
        except Exception:
            pass

        # Coverage tail: append a few generic top sources to ensure evaluator sees enough context
        tail_k = getattr(self.config, 'CITED_SOURCES_COVERAGE_TAIL', 2)
        if tail_k and tail_k > 0:
            generic = self.collect_sources(nodes)
            for g in generic:
                key = (g.get('file'), g.get('page_label'), g.get('id'))
                if key in seen:
                    continue
                sources.append(g)
                seen.add(key)
                if len([s for s in sources if s.get('file')]) >= len(cited) + tail_k:
                    break
        return sources or self.collect_sources(nodes)

    async def start_query_planning(self, event: StartQueryEvent):
        """Entry point for the query workflow."""
        print("\n--- Query Workflow Started ---")
        self._set_status('Planning query')

        self.context['rewrite_cycles'] = 0
        self.context['timings'] = {}
        self.context['source_refs'] = []
        chat_history = self.context.get('chat_history', [])
        self.context['workflow_start_time'] = time.monotonic()
        step_start_time = time.monotonic()

        try:
            plan = await self.query_planner_agent.aplan_query(event.query, chat_history=chat_history)
            self.context['timings']['query_planning'] = time.monotonic() - step_start_time

            # --- Sanitize rewritten query to avoid example bleedthrough ---
            import re as _re
            stopwords = {
                'the','a','an','and','or','but','if','then','else','for','to','of','in','on','at','by','with',
                'is','are','was','were','be','been','being','this','that','these','those','it','as','from','about',
                'what','which','who','whom','how','why','when','where','can','could','should','would','do','does','did'
            }
            def _tokens(s: str):
                return [t for t in _re.split(r"\W+", (s or '').lower()) if t and t not in stopwords and len(t) >= 4]

            original_q = event.query or ''
            planned_q = (plan.get("query") or '').strip()
            hyde_doc = (plan.get("hyde_document") or '').strip()
            orig_toks = set(_tokens(original_q))
            planned_toks = set(_tokens(planned_q))

            # If planned query shares no meaningful tokens with the original, fall back to original
            if not planned_q or (orig_toks and not (orig_toks & planned_toks)):
                print("[Planning Guard] Rewritten query appears unrelated; using original query instead.")
                planned_q = original_q
                # Drop HyDE if it doesn't align either
                hyde_toks = set(_tokens(hyde_doc))
                if orig_toks and not (orig_toks & hyde_toks):
                    hyde_doc = ''

            self.context['last_query'] = planned_q
            # Persist HyDE document for downstream writer guidance and fallback
            self.context['hyde_document'] = hyde_doc
        except Exception as exc:  # pragma: no cover - defensive guardrail
            self.context['timings']['query_planning'] = time.monotonic() - step_start_time
            self._set_status('Failed during planning')
            error_message = str(exc)
            print(f"Query planning failed: {error_message}")
            await self.dispatch(
                StopEvent(
                    result={
                        'error': error_message,
                        'phase': 'query_planning',
                        'timings': self.context.get('timings', {}),
                    }
                )
            )
            return

        # Persist retrieval decision in context for later gating/telemetry
        self.context['requires_retrieval'] = plan.get("requires_retrieval", True)
        await self.dispatch(
            QueryPlanningCompleteEvent(
                query=self.context['last_query'],
                hyde_document=self.context.get('hyde_document'),
                requires_retrieval=self.context['requires_retrieval'],
                result=plan
            )
        )
    async def start_search(self, ev: QueryPlanningCompleteEvent):
        """Triggered by QueryPlanningCompleteEvent. Runs the SearcherAgent."""
        # Local utility to score retrieval result quality consistently across branches
        def _retrieval_confidence(query_text: str, results):
            import re
            if not results:
                return 'low', {
                    'topk': 0, 'score_hits': 0, 'overlap_hits': 0,
                    'mean_overlap': 0.0, 'kb_hit': False
                }
            topk = min(getattr(self.config, 'RETR_CONF_TOPK', 5), len(results))
            top = results[:topk]
            # tokenize query: include long alpha tokens (>=3), ALL-CAPS acronyms (>=2),
            # numerical tokens, and alpha-numeric mixes like "r1", "gpt4o".
            stop = {
                'the','a','an','and','or','but','if','then','else','for','to','of','in','on','at','by','with',
                'is','are','was','were','be','been','being','this','that','these','those','it','as','from','about',
                'what','which','who','whom','how','why','when','where','can','could','should','would','do','does','did'
            }
            raw_q = query_text or ''
            split_tokens = [t for t in re.split(r"\W+", raw_q.lower()) if t]
            upper_tokens = re.findall(r"\b[A-Z]{2,}\b", raw_q)
            numeric_tokens = re.findall(r"\b\d+(?:\.\d+)?\b", raw_q)
            alnum_mix = re.findall(r"\b[a-zA-Z]*\d+[a-zA-Z0-9]*\b", raw_q)
            q_tokens_raw = [t for t in split_tokens if len(t) >= 3]
            q_tokens_raw.extend([t.lower() for t in upper_tokens])
            q_tokens_raw.extend([t.lower() for t in numeric_tokens])
            q_tokens_raw.extend([t.lower() for t in alnum_mix if len(t) >= 2])
            q_tokens = [t for t in q_tokens_raw if t not in stop]
            def _overlap(txt: str) -> int:
                toks = [t for t in re.split(r"\W+", (txt or '').lower()) if t]
                return len(set(q_tokens) & set(toks))
            overlaps = []
            score_hits = 0
            MIN_SCORE = getattr(self.config, 'RETR_CONF_MIN_SCORE', 0.25)
            for r in top:
                try:
                    txt = r.node.get_content() or ''
                except Exception:
                    txt = ''
                overlaps.append(_overlap(txt))
                try:
                    sc = float(r.score) if r.score is not None else 0.0
                except Exception:
                    sc = 0.0
                if sc >= MIN_SCORE:
                    score_hits += 1
            mean_overlap = (sum(overlaps)/len(overlaps)) if overlaps else 0.0
            MIN_OVERLAP = getattr(self.config, 'RETR_CONF_MIN_OVERLAP', 2)
            MIN_HITS = getattr(self.config, 'RETR_CONF_MIN_HITS', 2)
            overlap_hits = sum(1 for ov in overlaps if ov >= MIN_OVERLAP)
            # KB keyword cue: check if query mentions any filename tokens from results
            kb_tokens = set()
            for r in top:
                md = getattr(r.node, 'metadata', {}) or {}
                fn = (md.get('source_file') or md.get('file_name') or '')
                base = fn.rsplit('.', 1)[0]
                for tok in re.split(r"\W+", base.lower()):
                    if len(tok) >= 3:
                        kb_tokens.add(tok)
            kb_hit = any(tok in set(q_tokens) for tok in kb_tokens)

            MEAN_REQ = getattr(self.config, 'RETR_CONF_MEAN_OVERLAP', 2.0)
            if (kb_hit and (overlap_hits >= MIN_HITS or mean_overlap >= MEAN_REQ)) or (overlap_hits >= MIN_HITS and score_hits >= MIN_HITS):
                label = 'high'
            elif (overlap_hits == 0 and score_hits == 0 and not kb_hit):
                label = 'low'
            else:
                label = 'mid'
            return label, {
                'topk': topk,
                'score_hits': score_hits,
                'overlap_hits': overlap_hits,
                'mean_overlap': round(mean_overlap, 2),
                'kb_hit': kb_hit,
            }
        if not ev.requires_retrieval:
            # Soft probe with BM25 to prevent false negatives from the planner.
            probe_results = []
            bm25 = getattr(self.searcher_agent, 'bm25_retriever', None)
            if bm25 is not None:
                try:
                    bm25_q = await asyncio.to_thread(bm25.retrieve, ev.query)
                    if bm25_q:
                        probe_results.extend(bm25_q)
                except Exception as exc:
                    print(f"Probe BM25 (query) failed: {exc}")
                try:
                    if ev.hyde_document:
                        bm25_h = await asyncio.to_thread(bm25.retrieve, ev.hyde_document)
                        if bm25_h:
                            probe_results.extend(bm25_h)
                except Exception as exc:
                    print(f"Probe BM25 (HyDE) failed: {exc}")

            if not probe_results:
                # No evidence that retrieval could help; answer directly.
                print("\n--- Retrieval not required. Skipping Search. ---")
                self._set_status('Answering directly')
                # Mark confidence as 'none' for evaluation telemetry
                self.context['retrieval_confidence_label'] = 'none'
                self.context['retrieval_confidence_features'] = {}

                step_start_time = time.monotonic()
                final_answer = await self.direct_answer_agent.adirect_answer(
                    query=ev.query,
                    documents=ev.hyde_document
                )
                self.context['timings']['direct_answer'] = time.monotonic() - step_start_time
                self.context['timings']['total_workflow'] = time.monotonic() - self.context['workflow_start_time']
                self.context['source_refs'] = []
                self._set_status('Completed')
                await self.dispatch(StopEvent(result=self._build_result(
                    final_answer=final_answer,
                    verification_feedback="N/A (Direct Answer)"
                )))
                return
            else:
                # Compute confidence on probe and proceed to full retrieval unless clearly low.
                label, features = _retrieval_confidence(ev.query, probe_results)
                print(f"[RetrievalConfidence:probe] label={label} features={features}")
                self.context['retrieval_confidence_label'] = label
                self.context['retrieval_confidence_features'] = features
                if label == 'low':
                    print("\n--- Retrieval confidence LOW (probe); answering directly. ---")
                    self._set_status('Answering directly (planner & probe)')
                    step_start_time = time.monotonic()
                    final_answer = await self.direct_answer_agent.adirect_answer(
                        query=ev.query,
                        documents=ev.hyde_document
                    )
                    self.context['timings']['direct_answer'] = time.monotonic() - step_start_time
                    self.context['timings']['total_workflow'] = time.monotonic() - self.context['workflow_start_time']
                    self.context['source_refs'] = []
                    self._set_status('Completed')
                    await self.dispatch(StopEvent(result=self._build_result(
                        final_answer=final_answer,
                        verification_feedback=f"N/A (Planner+Confidence={label})"
                    )))
                    return

        step_start_time = time.monotonic()
        self._set_status('Searching knowledge base')
        search_results = await self.searcher_agent.asearch(
            query=ev.query,
            hyde_document=ev.hyde_document
        )
        self.context['timings']['search'] = time.monotonic() - step_start_time

        # Retrieval confidence gating (two-key): only skip downstream phases for low confidence.

        label, features = _retrieval_confidence(ev.query, search_results)
        print(f"[RetrievalConfidence] label={label} features={features}")
        # Persist for downstream result payloads
        self.context['retrieval_confidence_label'] = label
        self.context['retrieval_confidence_features'] = features
        if label == 'low':
            # Only skip retrieval on clearly low-confidence results.
            print("\n--- Retrieval confidence LOW; answering directly. ---")
            self._set_status('Answering directly (confidence gate)')
            step_start_time = time.monotonic()
            final_answer = await self.direct_answer_agent.adirect_answer(
                query=ev.query,
                documents=ev.hyde_document
            )
            self.context['timings']['direct_answer'] = self.context['timings'].get('direct_answer', 0) + (time.monotonic() - step_start_time)
            self.context['timings']['total_workflow'] = time.monotonic() - self.context['workflow_start_time']
            self.context['source_refs'] = []
            self._set_status('Completed')
            await self.dispatch(StopEvent(result=self._build_result(
                final_answer=final_answer,
                verification_feedback=f"N/A (Confidence={label})"
            )))
            return

        await self.dispatch(
            SearchCompleteEvent(
                query=ev.query,
                search_results=search_results
            )
        )

    async def start_reranking(self, ev: SearchCompleteEvent):
        """Triggered by SearchCompleteEvent. Runs the RerankerAgent."""
        step_start_time = time.monotonic()
        self._set_status('Reranking results')
        reranked_results = await self.reranker_agent.arerank(
            query=ev.query,
            documents=ev.search_results
        )
        self.context['timings']['reranking'] = time.monotonic() - step_start_time

        await self.dispatch(
            RerankCompleteEvent(
                query=ev.query,
                reranked_results=reranked_results
            )
        )

    async def handle_writing_request(self, ev: RerankCompleteEvent | RewriteEvent):
        """Handles both initial writing and rewrites."""
        feedback = ev.feedback if isinstance(ev, RewriteEvent) else None
        previous_answer = ev.previous_answer if isinstance(ev, RewriteEvent) else None
        unsupported_sentences = ev.unsupported_sentences if isinstance(ev, RewriteEvent) else None
        missing_citations = ev.missing_citations if isinstance(ev, RewriteEvent) else None
        required_facts = ev.required_facts if isinstance(ev, RewriteEvent) else None

        self.context['source_refs'] = self.collect_sources(ev.reranked_results)
        # Keep last reranked results for mapping-quality gating and packaging reuse
        self.context['last_reranked_results'] = ev.reranked_results

        phase_label = 'Rewriting answer' if feedback else 'Composing answer'
        self._set_status(phase_label)

        step_start_time = time.monotonic()
        generated_answer = await self.writer_agent.awrite_answer(
            query=ev.query,
            reranked_results=ev.reranked_results,
            feedback=feedback,
            previous_answer=previous_answer,
            guidance=self.context.get('hyde_document'),
            unsupported_sentences=unsupported_sentences,
            missing_citations=missing_citations,
            required_facts=required_facts
        )
        self.context['timings']['writing'] = self.context['timings'].get('writing', 0) + (time.monotonic() - step_start_time)

        await self.dispatch(
            WritingCompleteEvent(
                query=ev.query,
                reranked_results=ev.reranked_results,
                generated_answer=generated_answer
            )
        )

    async def start_verification(self, ev: WritingCompleteEvent):
        """Triggered by WritingCompleteEvent. Runs the VerifierAgent if enabled."""
        if not self.config.USE_VERIFIER:
            print("\n --- Verification skipped by configuration ---")
            self._set_status('Finalizing answer')
            # Build page-biased sources based on citations in the generated answer
            self.context['source_refs'] = self._page_biased_sources(ev.generated_answer, ev.reranked_results)
            self.context['timings']['total_workflow'] = time.monotonic() - self.context['workflow_start_time']
            self._set_status('Completed')
            await self.dispatch(StopEvent(result=self._build_result(
                final_answer=ev.generated_answer,
                verification_feedback="Verification was disabled"
            )))
            return

        # Gating: skip verification for clearly well-cited answers with available sources
        def _is_well_cited(answer: str) -> bool:
            if not answer:
                return False
            import re
            # Split into sentences conservatively
            # Keep non-empty lines, split by . ! ? while preserving bracketed citations
            raw_segs = re.split(r"(?<=[.!?])\s+", answer.strip())
            segs = [s.strip() for s in raw_segs if s and not s.isspace()]
            if not segs:
                return False
            # Allowed files from collected sources
            allowed_files = {(s.get('file') or '').strip() for s in (self.context.get('source_refs') or []) if s.get('file')}
            cited = 0
            for s in segs:
                # Consider a sentence cited if it ends with one or more bracketed citations like [file.pdf, p.X]
                if re.search(r"\]\s*$", s) and re.search(r"\[[^\]]+\]", s):
                    # Validate that all cited files are within the allowed set; otherwise do not count this sentence
                    ok = True
                    for (f, _p) in self._parse_citations(s):
                        if allowed_files and f not in allowed_files:
                            ok = False
                            break
                    if ok:
                        cited += 1
                else:
                    # Allow minimal uncited trailing sentence if it is boilerplate (e.g., closing remark)
                    pass
            # Require mapping quality: ensure page-biased mapping yields enough sources
            mapped_sources = self._page_biased_sources(answer, self.context.get('last_reranked_results') or [])
            # Bracket count (rough proxy for cited sentences)
            bracket_count = len(re.findall(r"\[[^\]]+\]", answer or ''))
            # At least half of cited brackets should map to sources, or at least 2
            has_mapping = bool(mapped_sources) and (len(mapped_sources) >= max(2, bracket_count // 2))
            return (cited >= max(1, len(segs) - 1)) and has_mapping

        has_sources = bool(self.context.get('source_refs'))
        if has_sources and _is_well_cited(ev.generated_answer):
            print("\n --- Verification gated: answer appears well-cited; skipping verifier ---")
            self._set_status('Completed')
            self.context['source_refs'] = self._page_biased_sources(ev.generated_answer, ev.reranked_results)
            self.context['timings']['total_workflow'] = time.monotonic() - self.context['workflow_start_time']
            await self.dispatch(StopEvent(result=self._build_result(
                final_answer=ev.generated_answer,
                verification_feedback="Skipped: well-cited answer"
            )))
            return

        step_start_time = time.monotonic()
        self._set_status('Verifying answer')
        verification_result = await self.verifier_agent.averify_answer(
            query=ev.query,
            generated_answer=ev.generated_answer,
            source_context=ev.reranked_results
        )
        self.context['timings']['verification'] = self.context['timings'].get('verification', 0) + (time.monotonic() - step_start_time)

        is_faithful = verification_result.get("is_faithful", False)
        if is_faithful:
            print("\n--- Answer is faithful. Workflow complete. ---")
            self.context['source_refs'] = self._page_biased_sources(ev.generated_answer, ev.reranked_results)
            self.context['timings']['total_workflow'] = time.monotonic() - self.context['workflow_start_time']
            self._set_status('Completed')
            await self.dispatch(StopEvent(result=self._build_result(
                final_answer=ev.generated_answer,
                verification_feedback=verification_result.get("feedback")
            )))
            return

        # Structured feedback handling
        unsupported = list(verification_result.get('unsupported_sentences') or [])
        missing = list(verification_result.get('missing_citations') or [])
        req_facts = list(verification_result.get('required_facts') or [])

        # If only missing citations are reported, avoid risky rewrites: perform evidence-anchored citation fill
        if not unsupported and missing:
            print("\n--- Verification indicates only missing citations; applying evidence-anchored citation fill. ---")


            from rank_bm25 import BM25Okapi  # preferred
            class BM25Okapi:  # type: ignore
                def __init__(self, tokenized_corpus):
                    from collections import Counter
                    self.corpus = tokenized_corpus or []
                    self.N = len(self.corpus)
                    self.df = Counter()
                    for doc in self.corpus:
                        for term in set(doc):
                            self.df[term] += 1
                    self.avgdl = (sum(len(doc) for doc in self.corpus) / self.N) if self.N else 1.0
                    self.k1 = 1.5
                    self.b = 0.75
                    self._idf_cache = {}
                def _idf(self, term):
                    n = self.df.get(term, 0)
                    return math.log(1 + (self.N - n + 0.5) / (n + 0.5)) if self.N else 0.0
                def get_scores(self, query_tokens):
                    scores = [0.0] * self.N
                    for idx, doc in enumerate(self.corpus):
                        dl = len(doc) or 1
                        tf = {}
                        for t in doc:
                            tf[t] = tf.get(t, 0) + 1
                        s = 0.0
                        for q in query_tokens or []:
                            f = tf.get(q, 0)
                            if f == 0:
                                continue
                            idf = self._idf_cache.get(q)
                            if idf is None:
                                idf = self._idf(q)
                                self._idf_cache[q] = idf
                            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                            s += idf * (f * (self.k1 + 1)) / denom
                        scores[idx] = s
                    return scores
            import re as _re

            def _tokenize(text: str) -> list[str]:
                return [t for t in _re.split(r"\W+", (text or '').lower()) if t and len(t) > 2]

            corpus_nodes = []
            corpus_tokens = []
            for nws in ev.reranked_results:
                node = nws.node
                corpus_nodes.append(node)
                corpus_tokens.append(_tokenize(node.get_content() or ''))
            bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

            def _clean_sentences(s: str) -> list[str]:
                segs = _re.split(r"(?<=[.!?])\s+", (s or '').strip())
                return [seg for seg in segs if seg and not seg.isspace()]

            def _best_node_for_sentence(sent: str):
                if not bm25 or not corpus_nodes:
                    return None
                query_tokens = _tokenize(sent)
                if not query_tokens:
                    return None
                scores = bm25.get_scores(query_tokens)
                if not scores:
                    return None
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                best_score = scores[best_idx]
                # Confidence heuristics: require token overlap and margin over next best
                # Compute token overlap between query and best node tokens
                try:
                    node_tokens = set(corpus_tokens[best_idx])
                except Exception:
                    node_tokens = set()
                overlap = len(set(query_tokens) & node_tokens)
                # Find second best score for margin check
                second_best = 0.0
                if len(scores) > 1:
                    # Avoid picking the same index
                    sorted_scores = sorted(((s, i) for i, s in enumerate(scores)), reverse=True)
                    for s, i in sorted_scores:
                        if i != best_idx:
                            second_best = s
                            break
                # Thresholds from configuration
                MIN_OVERLAP = getattr(self.config, 'CITATION_BM25_MIN_OVERLAP', 2)
                MARGIN_RATIO = getattr(self.config, 'CITATION_BM25_MARGIN_RATIO', 1.2)
                MIN_ABS = getattr(self.config, 'CITATION_BM25_MIN_ABS', 1.5)
                if overlap < MIN_OVERLAP:
                    return None
                # Accept if absolute score high or clearly above next best
                if best_score < MIN_ABS and not (second_best and best_score >= second_best * MARGIN_RATIO):
                    return None
                return corpus_nodes[best_idx]

            def _make_citation(node) -> str | None:
                meta = getattr(node, 'metadata', {}) or {}
                file = meta.get('source_file') or meta.get('file_name')
                page = meta.get('source_page_label') or meta.get('page_label')
                if file and page:
                    return f"[{file}, p.{page}]"
                if file:
                    return f"[{file}]"
                return None

            answer_sentences = _clean_sentences(ev.generated_answer)
            missing_set = {m.strip() for m in (missing or []) if m and m.strip()}
            patched_sentences: list[str] = []
            for s in answer_sentences:
                s_strip = s.strip()
                needs_cite = any(s_strip.endswith(t) or t in s_strip for t in missing_set)
                if needs_cite and not _re.search(r"\[[^\]]+\]\s*$", s_strip):
                    best_node = _best_node_for_sentence(s_strip)
                    cite = _make_citation(best_node) if best_node is not None else None
                    if cite:
                        if s_strip and s_strip[-1] in '.!?':
                            s_new = s_strip[:-1] + f" {cite}" + s_strip[-1]
                        else:
                            s_new = s_strip + f" {cite}"
                        patched_sentences.append(s_new)
                        continue
                patched_sentences.append(s_strip)
            patched_answer = ' '.join(patched_sentences)

            self.context['source_refs'] = self._page_biased_sources(patched_answer, ev.reranked_results)
            self.context['timings']['total_workflow'] = time.monotonic() - self.context['workflow_start_time']
            self._set_status('Completed')
            await self.dispatch(StopEvent(result=self._build_result(
                final_answer=patched_answer,
                verification_feedback=verification_result.get("feedback") or "Applied evidence-anchored citation fill."
            )))
            return

        self.context['rewrite_cycles'] += 1
        if self.context['rewrite_cycles'] >= self.config.MAX_REWRITES:
            print(f"\n--- Max rewrite limit ({self.config.MAX_REWRITES}) reached. Stopping. ---")
            self._set_status('Completed (verification failed)')
            self.context['source_refs'] = self._page_biased_sources(ev.generated_answer, ev.reranked_results)
            await self.dispatch(StopEvent(result=self._build_result(
                final_answer=ev.generated_answer,
                verification_feedback=f"FINAL ATTEMPT FAILED: {verification_result.get('feedback')}"
            )))
            return

        print(f"\n--- Answer not faithful. Starting rewrite cycle {self.context['rewrite_cycles']}. ---")
        self._set_status('Applying feedback')
        await self.dispatch(
            RewriteEvent(
                query=ev.query,
                reranked_results=ev.reranked_results,
                feedback=verification_result.get("feedback"),
                previous_answer=ev.generated_answer,
                unsupported_sentences=unsupported,
                missing_citations=missing,
                required_facts=req_facts,
            )
        )
