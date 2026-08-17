# Inspection report

- Working root: `D:\Rag_2\RAG\rag_foundation\buoi_14`
- Source (read-only): `D:\Rag_2\ner_kb`
- Note: `kb+hops` was absent; selected the complete `ner_kb` triplet after schema inspection.
- Python: `3.11.4`

## metadata.csv

- Rows: 30
- Columns: `id, title, so_ky_hieu, ngay_ban_hanh, loai_van_ban, ngay_co_hieu_luc, ngay_het_hieu_luc, nguon_thu_thap, ngay_dang_cong_bao, nganh, linh_vuc, co_quan_ban_hanh, chuc_danh, nguoi_ky, pham_vi, thong_tin_ap_dung, tinh_trang_hieu_luc`
- Encoding: utf-8
- Exact duplicate rows: 0
- Null counts: `{'id': 0, 'title': 0, 'so_ky_hieu': 0, 'ngay_ban_hanh': 0, 'loai_van_ban': 0, 'ngay_co_hieu_luc': 2, 'ngay_het_hieu_luc': 28, 'nguon_thu_thap': 10, 'ngay_dang_cong_bao': 22, 'nganh': 5, 'linh_vuc': 3, 'co_quan_ban_hanh': 0, 'chuc_danh': 1, 'nguoi_ky': 1, 'pham_vi': 0, 'thong_tin_ap_dung': 30, 'tinh_trang_hieu_luc': 0}`

## content.csv

- Rows: 30
- Columns: `id, content_html`
- Encoding: utf-8
- Exact duplicate rows: 0
- Null counts: `{'id': 0, 'content_html': 0}`

## relationships.csv

- Rows: 187
- Columns: `source, target, relationship_type, method, confidence, evidence`
- Encoding: utf-8-sig
- Exact duplicate rows: 0
- Null counts: `{'source': 0, 'target': 0, 'relationship_type': 0, 'method': 0, 'confidence': 0, 'evidence': 0}`

## Relationship types

- AP_DUNG_CHO: 81
- BAN_HANH_BOI: 31
- KY_BOI: 30
- SUA_DOI_BO_SUNG: 5
- THAM_CHIEU: 15
- THUOC_LINH_VUC: 25

## Safety

No source files are written. No destructive Neo4j statement is used.

Safe to continue: YES