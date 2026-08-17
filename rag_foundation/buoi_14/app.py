import html
import streamlit as st
from src.pipeline import RetrievalPipeline
from src.graph_hints import get_graph_hints
st.set_page_config(page_title="RAG Hybrid Search — Buổi 14",layout="wide")
st.title("RAG Hybrid Search — Buổi 14")
st.markdown("""
<style>
.lookup-card {border:1px solid #d8e2f1;border-radius:14px;padding:16px 18px;margin:12px 0 18px;background:#fff;box-shadow:0 2px 8px rgba(15,45,90,.05)}
.lookup-head {display:flex;gap:12px;align-items:flex-start;justify-content:space-between;margin-bottom:12px}
.citation-pill {display:inline-block;max-width:88%;padding:8px 12px;border:1px solid #9bc4ff;border-radius:18px;background:#eef6ff;color:#075dcc;font-size:.88rem;font-weight:600}
.rank-pill {min-width:78px;text-align:center;padding:7px;border-radius:18px;background:#e7f8eb;color:#14823b;font-size:.78rem;font-weight:700;line-height:1.35}
.lookup-text {color:#263238;font-size:.96rem;line-height:1.72;white-space:pre-wrap}
.result-summary {color:#596579;margin:-4px 0 14px}
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def pipeline(): return RetrievalPipeline()
def expanded_text(row, corpus):
 matches=[x["text"] for x in corpus if x["document_id"]==row["document_id"] and x.get("article")==row.get("article")]
 text="\n".join(dict.fromkeys(matches)).strip() if matches else row["text"]
 return text[:5000]+("…" if len(text)>5000 else "")
question=st.text_input("Câu hỏi")
label=st.selectbox("Method",["BM25","Dense","Hybrid","Hybrid + Rerank"]); top_k=st.slider("Top-k",1,10,5)
if st.button("Tìm kiếm",type="primary"):
 if not question.strip(): st.warning("Hãy nhập câu hỏi.")
 else:
  method={"BM25":"bm25","Dense":"dense","Hybrid":"hybrid","Hybrid + Rerank":"hybrid_rerank"}[label]
  rows=pipeline().retrieve(question,method,top_k)
  st.markdown(f"### 📌 Kết quả Tra cứu ({html.escape(label)})")
  st.markdown(f'<div class="result-summary">Hiển thị Top {len(rows)} kết quả cho câu hỏi: <em>“{html.escape(question)}”</em></div>',unsafe_allow_html=True)
  corpus=pipeline().hybrid.bm25.corpus
  for x in rows:
   score=x.get("score",x.get("retrieval_score",0.0))
   body=expanded_text(x,corpus)
   st.markdown(f'''<div class="lookup-card">
   <div class="lookup-head"><div class="citation-pill">📚 {html.escape(x['citation'])}</div>
   <div class="rank-pill">Rank #{x['rank']}<br>Score: {float(score):.4f}</div></div>
   <div class="lookup-text">{html.escape(body)}</div></div>''',unsafe_allow_html=True)
   with st.expander("Chi tiết kỹ thuật"):
    st.json({k:x.get(k) for k in ("chunk_id","document_id","retrieval_method","hybrid_rank","bm25_rank","dense_rank","rrf_score","rerank_logit") if x.get(k) is not None})
  st.subheader("Graph hints"); hints,status=get_graph_hints(rows);st.caption(status)
  st.dataframe(hints or [{"document_id":x["document_id"],"chunk_id":x["chunk_id"]} for x in rows])
  if method=="hybrid_rerank":
   st.subheader("Before / After Rerank")
   st.caption("Before = hạng trong tập Hybrid candidates; After = hạng Top-k sau Cross-Encoder. Relevance là sigmoid(logit), nằm trong khoảng 0–1.")
   st.dataframe([{"hybrid_rank_before":x.get("hybrid_rank"),"rerank_after":x["rank"],"chunk_id":x["chunk_id"],"relevance_0_1":x.get("rerank_score")} for x in rows])
