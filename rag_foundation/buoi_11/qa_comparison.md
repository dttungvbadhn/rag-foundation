# So sánh hệ thống QA theo số bước nhảy

- Thời điểm UTC: `2026-08-12T15:27:02.954379+00:00`
- Thiết lập cố định: `top_k=5`, cùng model embedding, vector index và prompt.
- Biến độc lập: `max_hops ∈ {0, 1, 2}`.
- Trạng thái: **NOT RUN / INCOMPLETE**.

## Kết quả định lượng

| Câu | Hop | Trạng thái | Seed | Related | Context chars | Quan hệ | Latency ms | Gemini calls |
|---|---:|---|---:|---:|---:|---|---:|---:|
| Q01 | 0 | answered | 5 | 0 | 1795 | — | 25298.938 | 1 |
| Q01 | 1 | answered | 5 | 0 | 1795 | — | 11304.76 | 1 |
| Q01 | 2 | answered | 5 | 0 | 1795 | — | 12488.858 | 1 |
| Q02 | 0 | answered | 5 | 0 | 2255 | — | 14744.583 | 1 |
| Q02 | 1 | answered | 5 | 1 | 2748 | HOP_NHAT | 18151.059 | 1 |
| Q02 | 2 | failed | — | — | — | — | 5462.055 | 0 |
| Q03 | 0 | answered | 5 | 0 | 2640 | — | 13769.332 | 1 |
| Q03 | 1 | failed | — | — | — | — | 5522.978 | 0 |
| Q03 | 2 | answered | 5 | 3 | 4498 | HOP_NHAT, VAN_BAN_BO_SUNG | 19937.987 | 1 |
| Q04 | 0 | answered | 5 | 0 | 2098 | — | 11404.687 | 1 |
| Q04 | 1 | answered | 5 | 1 | 2732 | CAN_CU | 10087.994 | 1 |
| Q04 | 2 | failed | — | — | — | — | 5458.83 | 0 |
| Q05 | 0 | failed | — | — | — | — | 5223.074 | 0 |
| Q05 | 1 | failed | — | — | — | — | 5327.152 | 0 |
| Q05 | 2 | failed | — | — | — | — | 5481.444 | 0 |

## Câu trả lời theo từng cấu hình

### Q01

Nghị định 46/2023/NĐ-CP thay thế cho nghị định nào, và nghị định bị thay thế đó có nội dung gì nổi bật về kinh doanh bảo hiểm?

**0 hop:** Không tìm thấy thông tin trong ngữ cảnh được cung cấp.

Ngữ cảnh hiện tại thiếu thông tin quy định về việc Nghị định 46/2023/NĐ-CP thay thế cho nghị định nào (điều khoản thi hành/thay thế) cũng như thông tin tổng thể về các nội dung nổi bật bị thay thế.

**1 hop:** Không tìm thấy thông tin trong ngữ cảnh được cung cấp.

**Thông tin còn thiếu:** Ngữ cảnh được cung cấp không chứa thông tin hoặc điều khoản quy định về việc Nghị định 46/2023/NĐ-CP thay thế cho nghị định nào, cũng như không cung cấp đủ thông tin chi tiết về các nội dung nổi bật của nghị định bị thay thế [S1], [S2], [S3], [S4], [S5].

**2 hop:** Không tìm thấy thông tin trong ngữ cảnh được cung cấp.

Cụ thể, ngữ cảnh hiện tại thiếu các thông tin sau:
- Quy định/điều khoản xác định Nghị định 46/2023/NĐ-CP bãi bỏ hoặc thay thế cho nghị định nào [S1].
- Nội dung chi tiết nổi bật về kinh doanh bảo hiểm của nghị định bị thay thế (ngữ cảnh chỉ bao gồm tiêu đề của Điều 85 Nghị định 46/2023/NĐ-CP [S1] và tiêu đề các Điều 53, 54, 55, 58 của Nghị định 73/2016/NĐ-CP [S2], [S3], [S4], [S5] mà không có nội dung quy định chi tiết hay quan hệ thay thế giữa các văn bản).

### Q02

Văn bản hợp nhất số 52/VBHN-NHNN được hợp nhất từ văn bản nào, và quy định về hồ sơ, thủ tục cấp giấy phép lần đầu của ngân hàng thương mại gồm những tài liệu gì?

**0 hop:** Không tìm thấy thông tin trong ngữ cảnh được cung cấp.

Cụ thể, ngữ cảnh còn thiếu các thông tin sau:
1. **Thông tin về các văn bản được hợp nhất:** Ngữ cảnh không chứa dữ liệu cho biết Văn bản hợp nhất số 52/VBHN-NHNN được hợp nhất từ những văn bản quy phạm pháp luật nào.
2. **Danh mục hồ sơ đầy đủ:** Ngữ cảnh không cung cấp toàn bộ quy định hay danh mục đầy đủ các tài liệu trong hồ sơ cấp Giấy phép lần đầu của ngân hàng thương mại, mà chỉ có một vài trích đoạn lẻ tẻ như:
   - Văn bản của ngân hàng thương mại nơi Ban trù bị mở tài khoản góp vốn xác nhận số tiền góp vốn của các cổ đông góp vốn thành lập [S3], [S4].
   - Văn bản của cấp có thẩm quyền chấp thuận cho góp vốn thành lập ngân hàng đối với trường hợp tổ chức là doanh nghiệp Nhà nước [S5].

**1 hop:** Không tìm thấy thông tin trong ngữ cảnh được cung cấp.

Cụ thể, ngữ cảnh còn thiếu các thông tin sau:

1. **Thông tin về các văn bản cấu thành nên Văn bản hợp nhất số 52/VBHN-NHNN:** Ngữ cảnh không nêu rõ danh sách các văn bản nguồn được hợp nhất để tạo thành văn bản này (chỉ ghi nhận quan hệ hợp nhất với Thông tư số 56/2024/TT-NHNN qua đường đi đồ thị [R1]).
2. **Danh mục đầy đủ hồ sơ, thủ tục cấp Giấy phép lần đầu của ngân hàng thương mại:** Ngữ cảnh chỉ đề cập đến một số tài liệu đơn lẻ trong hồ sơ, chưa có quy định toàn diện:
   - Văn bản của ngân hàng thương mại nơi Ban trù bị mở tài khoản góp vốn xác nhận số tiền góp vốn của các cổ đông góp vốn thành lập [S3], [S4].
   - Văn bản của cấp có thẩm quyền chấp thuận cho góp vốn thành lập ngân hàng đối với trường hợp tổ chức là doanh nghiệp Nhà nước [S5], [R1].

**2 hop:** NOT RUN — ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash\nPlease retry in 6.423403671s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.6-flash'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '6s'}]}}

### Q03

Thông tư số 01/2025/TT-NHNN quy định về cấp giấy phép quỹ tín dụng nhân dân được sửa đổi, bổ sung bởi văn bản nào, và những nội dung sửa đổi bổ sung chính là gì?

**0 hop:** Dựa trên dữ liệu ngữ cảnh được cung cấp:

1. **Văn bản sửa đổi, bổ sung:**
   Thông tư số 01/2025/TT-NHNN quy định về cấp Giấy phép lần đầu, cấp đổi Giấy phép của quỹ tín dụng nhân dân được sửa đổi, bổ sung bởi **Thông tư số 63/2025/TT-NHNN** (quy định tại Chương II) [S1].

2. **Các nội dung sửa đổi, bổ sung chính:**
   Không tìm thấy thông tin trong ngữ cảnh được cung cấp. 
   *(Thông tin còn thiếu: Chi tiết từng điều khoản và nội dung sửa đổi, bổ sung cụ thể của Thông tư số 01/2025/TT-NHNN trong Thông tư số 63/2025/TT-NHNN không xuất hiện trong đoạn trích)*.

**1 hop:** NOT RUN — ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash\nPlease retry in 47.156894894s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.6-flash'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '47s'}]}}

**2 hop:** Theo thông tin từ ngữ cảnh được cung cấp:

**1. Văn bản sửa đổi, bổ sung:**
Thông tư số 01/2025/TT-NHNN quy định về cấp Giấy phép lần đầu, cấp đổi Giấy phép của quỹ tín dụng nhân dân được sửa đổi, bổ sung bởi **Thông tư số 63/2025/TT-NHNN** (quy định tại Chương II) [S1].

**2. Nội dung sửa đổi, bổ sung chính:**
Không tìm thấy thông tin trong ngữ cảnh được cung cấp. Ngữ cảnh chỉ đề cập đến tiêu đề Chương II của Thông tư số 63/2025/TT-NHNN về việc sửa đổi, bổ sung Thông tư số 01/2025/TT-NHNN [S1], nhưng không có dữ liệu chi tiết về các nội dung hay điều khoản được sửa đổi, bổ sung cụ thể.

*Lưu ý: Đây là thông tin tra cứu dựa trên văn bản được cung cấp, không phải tuyên bố tư vấn pháp lý.*

### Q04

Thông tư số 41/2016/TT-NHNN về tỷ lệ an toàn vốn của ngân hàng căn cứ vào luật nào, và luật đó quy định chức năng nhiệm vụ của cơ quan nào?

**0 hop:** Không tìm thấy thông tin trong ngữ cảnh được cung cấp.

Cụ thể, ngữ cảnh hiện tại không chứa các đoạn trích liên quan đến Thông tư số 41/2016/TT-NHNN, các luật làm căn cứ cho Thông tư này, cũng như quy định về chức năng, nhiệm vụ của cơ quan liên quan.

**1 hop:** Không tìm thấy thông tin trong ngữ cảnh được cung cấp.

Hiện tại, ngữ cảnh không chứa bất kỳ thông tin nào về Thông tư số 41/2016/TT-NHNN, văn bản luật làm căn cứ cho Thông tư này, cũng như quy định về chức năng, nhiệm vụ của cơ quan liên quan trong luật đó.

**2 hop:** NOT RUN — ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash\nPlease retry in 242.139043ms.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.6-flash'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '0s'}]}}

### Q05

Hoạt động giao nhận, vận chuyển tiền mặt và tài sản quý của Ngân hàng Nhà nước được điều chỉnh bởi Thông tư nào, và Thông tư đó có được sửa đổi bổ sung bởi văn bản nào không?

**0 hop:** NOT RUN — ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash\nPlease retry in 55.01753155s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.6-flash'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '55s'}]}}

**1 hop:** NOT RUN — ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash\nPlease retry in 49.679969793s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-3.6-flash'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '49s'}]}}

**2 hop:** NOT RUN — ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-3.6-flash\nPlease retry in 44.203081619s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'model': 'gemini-3.6-flash', 'location': 'global'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '44s'}]}}

## Kết luận

Chưa thể chứng minh hiệu quả multi-hop bằng kết quả thực nghiệm vì ít nhất một cấu hình không chạy hoàn chỉnh. Không sử dụng dữ liệu giả để thay thế kết quả runtime.

Các câu trả lời và quan hệ thu được vẫn cần chuyên gia kiểm tra; số context lớn hơn không tự động đồng nghĩa chính xác hơn.
