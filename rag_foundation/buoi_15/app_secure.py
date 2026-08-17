import html
import streamlit as st
from src.secure_retriever import SecureRetriever
from src.security import VALID_ROLES

st.set_page_config(page_title="Secure RAG RBAC — Buổi 15",layout="wide")
st.markdown("""<style>.secure-card{border:1px solid #dbe5f2;border-radius:14px;padding:16px;margin:12px 0;background:white}.security-tag{display:inline-block;background:#fff3cd;color:#745600;border-radius:12px;padding:5px 10px;font-size:.82rem}.cite{color:#075dcc;font-weight:650}.body{line-height:1.65;color:#263238}</style>""",unsafe_allow_html=True)
st.title("🔐 Secure RAG — RBAC Buổi 15")

@st.cache_resource
def engine(): return SecureRetriever()

with st.sidebar:
 st.header("Cấu hình truy cập")
 roles=st.multiselect("Vai trò của bạn (Your Roles)",VALID_ROLES,default=["Guest"])
 label=st.selectbox("Method",["BM25","Dense","Hybrid","Hybrid + Rerank"],index=2)
 top_k=st.slider("Top-k",1,10,5);candidate_k=st.slider("Candidate-k",5,50,20)

question=st.text_input("Câu hỏi",placeholder="Nhập nội dung cần tra cứu...")
if st.button("Tìm kiếm an toàn",type="primary"):
 if not roles: st.error("Hãy chọn ít nhất một vai trò.")
 elif not question.strip(): st.warning("Hãy nhập câu hỏi.")
 else:
  method={"BM25":"bm25","Dense":"dense","Hybrid":"hybrid","Hybrid + Rerank":"hybrid_rerank"}[label]
  with st.spinner("Đang lọc quyền và tìm kiếm..."):
   rows=engine().retrieve(question,roles,method,top_k,candidate_k);hidden=engine().filtered_count(roles)
  st.success(f"Đã lọc bỏ {hidden:,} chunk do không đủ quyền truy cập. Trả về {len(rows)} kết quả hợp lệ.")
  st.markdown(f"### 📌 Kết quả Tra cứu ({html.escape(label)})")
  st.caption(f'Vai trò: {", ".join(roles)} · Câu hỏi: “{question}”')
  for x in rows:
   allowed=", ".join(x["allowed_roles"]);score=float(x.get("score",0))
   st.markdown(f'''<div class="secure-card"><div class="cite">📚 {html.escape(x['citation'])}</div><p><span class="security-tag">🔒 Quyền xem: {html.escape(allowed)}</span> &nbsp; Rank #{x['rank']} · Score {score:.4f}</p><div class="body">{html.escape(x['text'])}</div></div>''',unsafe_allow_html=True)
  st.subheader("Graph hints đã lọc quyền")
  hints=engine().graph_search(roles,top_k);st.dataframe(hints,use_container_width=True) if hints else st.info("Không có Graph hints hợp lệ hoặc Neo4j chưa sẵn sàng.")
  if method=="hybrid_rerank":
   st.subheader("Before / After Rerank")
   st.dataframe([{"hybrid_rank_before":x.get("hybrid_rank"),"rerank_after":x["rank"],"chunk_id":x["chunk_id"],"relevance_0_1":x.get("rerank_score")} for x in rows],use_container_width=True)
