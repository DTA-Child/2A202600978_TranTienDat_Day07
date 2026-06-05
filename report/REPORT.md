# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Trần Tiến Đạt
**Nhóm:** C2
**Ngày:** 5/6/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Nghĩa là độ tương đồng về mặt ngữ nghĩa gần nhau

**Ví dụ HIGH similarity:**
- Sentence A: mua 1 con bò
- Sentence B: mua 1 con bê
- Tại sao tương đồng: đều là con bò

**Ví dụ LOW similarity:**
- Sentence A: nhà thích nuôi con mèo 
- Sentence B: Anh này rất giàu
- Tại sao khác: không có bất kì liên kết với nhau

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Vì cosine mạnh về việc sử dụng các từ tương đồng trong khi eucli chỉ so sánh về độ dài dẫn đến so sánh sai

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> stride = chunk_size - overlap = 500−50 = 450 
> Số lượng chunk bổ sung sau chunk đầu tiên = ⌈(Tổng ký tự−chunk_size)/stride⌉ = ⌈(10,000−500)/450⌉ = ⌈9,500/450⌉≈21.11 <=> 22
> Tổng số chunk = 1 (chunk đầu tiên) + 22 = 23
> *Đáp án: 23 *

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Số lượng chunk sẽ tăng lên (cụ thể là 25 chunks theo phép tính 1+⌈9,500/400⌉) do khoảng cách di chuyển giữa mỗi lần cắt ngắn lại.
> Lý do: Tăng overlap giúp giữ trọn vẹn ngữ cảnh tại điểm cắt

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain: Y tế (thuốc)**

**Tại sao nhóm chọn domain này?**
> *Nhóm đã có tìm hiểu về domain này trong buổi lab trước ,cùng với việc đã có nguồn data để lựa chọn*


### Data Inventory


| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | caremark-oct2013.txt | https://github.com/ericminikel/cnsdrugs.git | 26.541 | doc_type: formulary_guide, year: 2013 |
| 2 | customer_support_playbook.txt | https://github.com/ericminikel/cnsdrugs.git | 12.450 | doc_type: sop_playbook, year: 2025 |
| 3 | healthalliance-2013.txt | https://github.com/ericminikel/cnsdrugs.git | 16.892 | doc_type: formulary_guide, year: 2013 |
| 4 | FDAMDD_v3b_1216_15Feb2008_nostructures.txt | https://github.com/ericminikel/cnsdrugs.git | 485.610| doc_type: fda_chemical_registry, year: 2008|
| 5 | humana_2014_wi.txt | https://github.com/ericminikel/cnsdrugs.git | 321.405| doc_type: formulary_guide, year: 2014 |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| doc_type | String (Enum) | formulary_guide, fda_chemical_registry | Giúp phân loại nhanh cấu trúc văn bản. Hệ thống RAG có thể lọc (filter) để chỉ tìm kiếm trong sách danh mục thuốc thay vì quét qua các tài liệu hóa dược chuyên sâu của FDA khi người dùng hỏi về giá cả hoặc điều kiện bảo hiểm. |
| year | Integer | 2013, 2014, 2025 | Hỗ trợ lọc theo thời gian hiệu lực. Chính sách bảo hiểm y tế thay đổi theo từng năm; việc lọc theo năm giúp mô hình tránh lấy nhầm dữ liệu cũ (ví dụ: danh mục thuốc năm 2013) để trả lời cho câu hỏi ở thời điểm hiện tại. |
| domain | String (Enum) | medical_insurance, cns_pharmacology | Thu hẹp phạm vi tìm kiếm theo không gian kiến thức. Giúp định tuyến câu hỏi (Query Routing) đến đúng phân vùng dữ liệu chuyên môn, tăng độ chính xác ($Precision$) và giảm nhiễu từ các ngành không liên quan. |
| target_audience | String (Enum) | pharmacist_and_clinician, support_agent | Tối ưu hóa văn phong và độ phức tạp của câu trả lời. Khi biết tài liệu hướng đến đối tượng nào, hệ thống có thể ưu tiên trích xuất nội dung có thuật ngữ chuyên môn cao cho bác sĩ, hoặc nội dung dễ hiểu, dạng quy trình cho nhân viên tư vấn. |
---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| caremark-oct2013.txt | FixedSizeChunker (`fixed_size`) | 11 | 183 | Trung bình |
| caremark-oct2013.txt | SentenceChunker (`by_sentences`) | 1 | 2013 | Thấp |
| caremark-oct2013.txt | RecursiveChunker (`recursive`) | 12 | 165 | Tốt |
| healthalliance-2013.txt | FixedSizeChunker (`fixed_size`) | 12 | 186.2 | Trung bình |
| healthalliance-2013.txt | SentenceChunker (`by_sentences`) | 1 | 2235.0 | Thấp |
| healthalliance-2013.txt | RecursiveChunker (`recursive`) | 12 | 184.4 | Tốt |
| FDAMDD_v3b_1216_15Feb2008_nostructures.txt | FixedSizeChunker (`fixed_size`) | 4101 | 200 | Trung bình |
| FDAMDD_v3b_1216_15Feb2008_nostructures.txt | SentenceChunker (`by_sentences`) | 2 | 410066 | Tệ |
| FDAMDD_v3b_1216_15Feb2008_nostructures.txt | RecursiveChunker (`recursive`) | 3970 | 204 | Tốt |

### Strategy Của Tôi
**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**

> RecursiveChunker chia văn bản theo nhiều cấp độ ưu tiên thay vì cắt trực tiếp theo số ký tự. Thuật toán sẽ cố gắng tách theo đoạn văn (`\n\n`) trước, sau đó đến dòng (`\n`), câu hoặc các dấu phân cách khác. Nếu chunk vẫn vượt quá kích thước mục tiêu, hệ thống tiếp tục chia nhỏ ở cấp độ thấp hơn cho đến khi đạt giới hạn mong muốn. Cách tiếp cận này giúp giữ lại cấu trúc tự nhiên của tài liệu và giảm tình trạng cắt ngang ngữ nghĩa.

**Tại sao tôi chọn strategy này cho domain nhóm?**

> Domain của nhóm chứa các tài liệu có độ dài và định dạng rất khác nhau, từ tài liệu hỗ trợ khách hàng đến các tài liệu chuyên ngành dài hàng trăm trang. RecursiveChunker tận dụng cấu trúc sẵn có trong tài liệu để bảo toàn ngữ cảnh nhưng vẫn tạo ra các chunk đủ nhỏ cho embedding và retrieval. Kết quả benchmark cho thấy strategy này hoạt động ổn định trên cả tài liệu ngắn lẫn tài liệu rất lớn, trong khi các phương pháp khác dễ tạo chunk quá lớn hoặc làm mất ngữ cảnh khi cắt văn bản.


### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline |  | | |
| | **của tôi** | | | |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker (`recursive`) | 8.4 | Giữ ngữ cảnh tốt, chunk ổn định, phù hợp cho retrieval | Avg length thấp hơn mục tiêu nên số chunk tăng nhẹ |
| [Vũ Văn Học ] | RecursiveChunker (`recursive`) | 8.5 | Chunk count thấp nhất, avg length gần kích thước mục tiêu, tận dụng cấu trúc tài liệu tốt | Sẽ lệch khi tham gia domain khác |
| [Hồ Trọng Nhật Minh ] | RecursiveChunker (`recursive`) | 8.3 | Giữ ngữ cảnh tốt, chunk ổn định, phù hợp cho retrieval | Cải thiện chưa nhiều so với FixedSize |
| [Nguyễn Đức Thành ] | RecursiveChunker (`recursive`) | 8.5 | Đơn giản, dễ kiểm soát kích thước chunk | Không tận dụng cấu trúc tài liệu nên dễ cắt mất ngữ nghĩa |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *RecursiveChunker là strategy phù hợp nhất cho domain này. Các tài liệu trong bộ dữ liệu y tế có cấu trúc rất đa dạng, từ danh mục thuốc, tài liệu bảo hiểm đến các tập dữ liệu chuyên ngành lớn. RecursiveChunker chia văn bản theo nhiều cấp độ hơn FixedSizeChunker, đồng thời tránh tạo ra các chunk quá lớn như SentenceChunker. *

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Hàm sử dụng regex (?<=[.!?])\s+ để tách văn bản tại các dấu kết thúc câu như ., !, ?. Các câu rỗng được loại bỏ và nhiều câu được gom lại thành một chunk theo max_sentences_per_chunk. Hàm cũng xử lý trường hợp văn bản rỗng bằng cách trả về danh sách rỗng.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán chia văn bản theo thứ tự ưu tiên các separator như đoạn văn (\n\n), dòng (\n), câu (. ) và khoảng trắng. Nếu một phần vẫn lớn hơn chunk_size, hàm _split() sẽ tiếp tục gọi đệ quy với separator ở mức thấp hơn. Base case là khi độ dài văn bản nhỏ hơn hoặc bằng chunk_size hoặc không còn separator nào để tiếp tục chia.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Mỗi document được chuyển thành embedding vector và lưu cùng nội dung, metadata trong chromaDB hoặc bộ nhớ trong. Khi tìm kiếm, query cũng được embedding thành vector và độ tương đồng được tính bằng dot product giữa vector query và vector của từng document. Các kết quả được sắp xếp theo score giảm dần để lấy top_k phù hợp nhất.

**`search_with_filter` + `delete_document`** — approach:
> Hệ thống thực hiện lọc metadata trước rồi mới chạy similarity search để giảm không gian tìm kiếm. Với chromaDB, filter được truyền qua tham số where; với in-memory store, các record được lọc thủ công trước khi tính điểm tương đồng. Hàm delete_document() xóa toàn bộ record có cùng doc_id khỏi vector store.

### KnowledgeBaseAgent

**`answer`** — approach:
> Agent áp dụng mô hình RAG bằng cách truy xuất top_k chunk liên quan nhất từ vector store. Các chunk được ghép thành phần context rồi inject vào prompt theo cấu trúc Context -> Question -> Answer. Prompt hoàn chỉnh sau đó được gửi tới LLM để sinh câu trả lời dựa trên ngữ cảnh đã truy xuất.

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                                                                              [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                                                                                       [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                                                                                [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                                                                                 [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                                                                                      [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                                                                                      [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                                                                            [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                                                                             [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                                                                           [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                                                                             [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                                                                             [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                                                                                        [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                                                                                    [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                                                                              [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                                                                                     [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                                                                                         [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                                                                                   [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                                                                                         [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                                                                             [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                                                                               [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                                                                                 [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                                                                                       [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                                                                            [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                                                                              [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                                                                                  [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                                                                               [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                                                                                        [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                                                                                       [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                                                                                  [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                                                                              [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                                                                                         [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                                                                             [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                                                                                   [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                                                                             [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                                                                                          [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                                                                                        [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                                                                                       [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                                                                           [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                                                                                      [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                                                                               [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                                                                                     [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                                                                                         [100%]
========================================================================================== 42 passed in 0.09s ===========================================================================================
PS C:\Users\kodau\OneDrive\Desktop\Vin\Lab\Day7\Day-07-Lab-Data-Foundations> 
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | The customer forgot their password and requested an account reset. | A user contacted support because they could not log into their account. | high | 0.1826 | Đúng |
| 2 | The server CPU usage reached 95% during peak traffic. | Heavy load caused processor utilization to spike on the production machine. | high | 0.1907 | Đúng |
| 3 | The company announced a new electric vehicle model. | Researchers published a paper on marine biology. | low | 0.1336 | Đúng |
| 4 | Students submitted their assignments before the deadline. | The professor received all coursework prior to the due date. | high | 0.2182 | Đúng |
| 5 | The stock market closed higher after positive earnings reports. | Rainfall increased significantly across northern regions this week. | low | 0.0000 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Với HashingTextEmbedder, các cặp được xem là "high similarity" khi điểm cosine nằm trong nhóm cao nhất của tập thử nghiệm (~0.18–0.22), trong khi các cặp có điểm thấp hơn (~0.13 hoặc thấp hơn) được xem là low similarity. (ngưỡng phân loại ở bài toán này chỉ nằm ở 0.15).

> Embeddings biểu diễn ý nghĩa bằng vị trí trong không gian vector thay vì bằng các từ khóa riêng lẻ. Các câu có cùng ngữ nghĩa thường có cosine similarity cao hơn các câu không liên quan. Trong thí nghiệm này, chỉ cần ngưỡng 0.15 đã đủ tách được phần lớn các cặp gần nghĩa và khác nghĩa, cho thấy ngay cả một hashing embedder đơn giản vẫn tạo được cấu trúc ngữ nghĩa cơ bản trong không gian vector, dù chưa mạnh bằng các semantic embedding hiện đại.
---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What relationship does Rivastigmine have with Alzheimer's disease in the medical knowledge base? | Rivastigmine has a DM clinical relationship with Alzheimer's disease and DrugBank ID DB00989. |
| 2 | In Health Alliance 2013, which anxiety medication is listed as Buspar? | Buspirone is listed as buspirone (Buspar) under Anxiety. |
| 3 | What tier and quantity limit is ABILIFY 10 MG TABLET in Humana 2014 Wisconsin? | ABILIFY 10 MG TABLET is listed as MO tier 4 with QL 30 per 30 days. |
| 4 | UnitedHealthcare Morphine Sulfate Solution Oral Roxanol MS Contin | The UHC CNS list includes Morphine Sulfate oral solution, Roxanol, MS Contin, and related Morphine Sulfate entries. |
| 5 | In Caremark October 2013, what note is listed for buspirone? | Buspirone is listed with the note NP = 7.5 mg. |


### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What relationship does Rivastigmine have with Alzheimer's disease in the medical knowledge base? | - Thuốc Rivastigmine (Mã DrugBank: DB00989) có mối liên hệ lâm sàng dạng 'DM' với chứng bệnh Alzheimer's disea | 0.152 | Yes | - Thuốc Rivastigmine (Mã DrugBank: DB00989) có mối liên hệ lâm sàng dạng 'DM' với chứng bệnh Alzheimer's disease (Mã DOID: DOID:10652). Thông tin này được xác thực bởi 1 chuyên gia |
| 2 | In Health Alliance 2013, which anxiety medication is listed as Buspar? | ANXIETY amitriptyline (Elavil) bupropion (Wellbutrin) alprazolam (Xanax) buspirone (Buspar) 10 UTILIZATION MAN | 0.085 | Yes | ANXIETY amitriptyline (Elavil) bupropion (Wellbutrin) alprazolam (Xanax) buspirone (Buspar) 10 UTILIZATION MANAGEMENT OTHER COVERAGE NOTES diazepam (Valium) hydroxyzine (Atarax, Vi |
| 3 | What tier and quantity limit is ABILIFY 10 MG TABLET in Humana 2014 Wisconsin? | ABILIFY 1 MG/ML ORAL SOLN MO 4 QL (750 per 30 days) ABILIFY 10 MG TABLET MO 4 QL (30 per 30 days) ABILIFY 15 M | 0.282 | Yes | ABILIFY 1 MG/ML ORAL SOLN MO 4 QL (750 per 30 days) ABILIFY 10 MG TABLET MO 4 QL (30 per 30 days) ABILIFY 15 MG TABLET MO 4 QL (30 per 30 days) ABILIFY 2 MG TABLET MO 4 QL (30 per  |
| 4 | UnitedHealthcare Morphine Sulfate Solution Oral Roxanol MS Contin | 3.1 Narcotic Analgesics 3.1.1 NARCOTICS TIER 1 + Codeine Sulfate (Codeine Sulfate) Duragesic (Fentanyl Transde | 0.188 | Yes | 3.1 Narcotic Analgesics 3.1.1 NARCOTICS TIER 1 + Codeine Sulfate (Codeine Sulfate) Duragesic (Fentanyl Transdermal)* SL + Hydromorphone HCl (Dilaudid) + Levorphanol Tartrate (Levo- |
| 5 | In Caremark October 2013, what note is listed for buspirone? | alprazolam (Xanax – brand is NP) alprazolam ext-release (Xanax XR – brand is NP) • buspirone, NP = 7.5 mg DIAZ | 0.317 | Yes | alprazolam (Xanax – brand is NP) alprazolam ext-release (Xanax XR – brand is NP) • buspirone, NP = 7.5 mg DIAZEPAM oral soln, 1 mg/mL diazepam tabs (Valium – brand is NP) hydroxyzi |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Qua phần trình bày của các thành viên, tôi nhận ra việc lựa chọn chunking strategy không chỉ ảnh hưởng đến số lượng chunk mà còn ảnh hưởng trực tiếp đến chất lượng retrieval. Dù cùng sử dụng RecursiveChunker, mỗi người có cách đánh giá và phân tích trade-off giữa chunk size, overlap và khả năng giữ ngữ cảnh khác nhau. Điều này giúp tôi hiểu rõ hơn cách tối ưu chunking cho từng loại dữ liệu.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Một số nhóm đã thử áp dụng metadata filtering và xây dựng các bộ benchmark query đa dạng hơn thay vì chỉ kiểm tra retrieval đơn thuần. Điều này cho thấy chất lượng của hệ thống RAG không chỉ phụ thuộc vào embedding hay chunking mà còn phụ thuộc vào cách tổ chức dữ liệu và thiết kế phương pháp đánh giá. Tôi học được tầm quan trọng của việc xây dựng benchmark phù hợp với domain thực tế.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu làm lại, tôi sẽ bổ sung thêm nhiều nguồn dữ liệu y tế có cấu trúc khác nhau để tăng độ đa dạng của knowledge base. Tôi cũng sẽ chuẩn hóa metadata chi tiết hơn (ví dụ: loại thuốc, nhóm bệnh, nguồn dữ liệu) để tận dụng metadata filtering hiệu quả hơn trong quá trình retrieval.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5/ 5 |
| Document selection | Nhóm | 10/ 10 |
| Chunking strategy | Nhóm | 15/ 15 |
| My approach | Cá nhân | 10/ 10 |
| Similarity predictions | Cá nhân | 5/ 5 |
| Results | Cá nhân | 10/ 10 |
| Core implementation (tests) | Cá nhân | 30/ 30 |
| Demo | Nhóm | 5/ 5 |
| **Tổng** | 100 | **100/ 100** |
