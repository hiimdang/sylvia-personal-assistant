# Sylvia - AI Personal Assistant

Sylvia là một chatbot RAG đa phương tiện được xây dựng với FastAPI, LangChain, Qdrant và FastEmbed.

## Highlights

- **Chat đa phương tiện**: Xử lý text và hình ảnh với trích xuất ngữ cảnh thông minh
- **RAG pipeline**: Trích xuất từ khóa + tìm kiếm vector + re-ranking kết quả
- **Quản lý phiên**: Lưu trữ bộ nhớ theo người dùng (in-memory)
- **Công cụ tìm kiếm web**: Hỗ trợ Brave / Google
- **Kiến trúc module hoá**: Cho phép thay thế linh hoạt vector database, embedding model và search provider mà không ảnh hưởng logic core

## Tech Stack

- FastAPI + Uvicorn
- LangChain
- Qdrant
- FastEmbed
- SentenceTransformers CrossEncoder
- Pydantic + pydantic-settings

## Project Structure

```text
semory_sylvia/
├── app.py
├── run.py
├── requirements.txt
├── .env.example
├── demo/
├── sylvia_core/
│   ├── config.py
│   ├── clients.py
│   ├── component_manager.py
│   ├── bootstrap.py
│   ├── ingest.py
│   ├── rag_core/
│   │   ├── rag_core.py
│   │   ├── retrieval_pipeline.py
│   │   ├── memory_manager.py
│   │   └── agent_builder.py
│   ├── providers/
│   │   ├── search_provider.py
│   │   ├── brave_search_provider.py
│   │   ├── google_search_provider.py
│   │   ├── fastembed_embedding.py
│   │   ├── qdrant_vector_store.py
│   │   ├── embedding_provider.py
│   │   └── vector_store_provider.py
│   ├── prompts/
│   │   ├── sylvia_persona_prompt_example.py
│   │   ├── retrieved_context_policy_example.py
│   │   ├── describe_image_prompt_example.py
│   │   ├── keyword_extraction_prompt_example.py
│   │   └── keyword_image_extraction_prompt_example.py
│   └── utils/
│       ├── extractors/
│       ├── retrieval/
│       ├── schemas/
│       ├── prompts/
│       ├── formatters/
│       ├── clients/
│       │   └── brave_search_client.py
│       └── web/
│           ├── web_tools_manager.py
│           └── search_tools.py
└── setup.py
```
## Runtime Flow

1. `app.py` nhận request từ endpoint `/chat`.
2. `SylviaRAGCore.get_response()` điều phối toàn bộ quá trình xử lý request.
3. `RetrievalPipeline.search()` thực hiện trích xuất từ khóa và tạo embedding song song.
4. Retriever truy vấn Qdrant và rerank kết quả.
5. Context cùng lịch sử hội thoại được tổng hợp thành messages.
6. Agent sinh phản hồi bằng LLM chính và có thể sử dụng thêm tools nếu cần.
7. Memory được cập nhật; tính năng ghi log tương tác là tùy chọn và mặc định bị tắt.

---

## Cấu hình

File cấu hình chính nằm tại `sylvia_core/config.py` (env prefix: `SYLVIA_`).

### Bắt buộc

- `SYLVIA_QDRANT_URL`
- `SYLVIA_QDRANT_API_KEY`
- `OPENAI_API_KEY`

### Tùy chọn

- `SYLVIA_SEARCH_PROVIDER_TYPE` (`brave`, `google`, `none`)
- `SYLVIA_BRAVE_SEARCH_API_KEY` (bắt buộc nếu provider là `brave`)
- `SYLVIA_GOOGLE_API_KEY` và `SYLVIA_GOOGLE_CSE_ID` (bắt buộc nếu provider là `google`)
- `SYLVIA_ENABLE_INTERACTION_LOG` (`false` mặc định)
- `SYLVIA_LOG_FILE` (chỉ dùng khi bật interaction logging)

Sử dụng `.env.example` làm template cấu hình.

---

## Chạy local

```bash
python run.py
```

Server mặc định:

- `http://127.0.0.1:6969`

---

## Ghi chú về Search

Hệ thống hiện sử dụng dependency injection theo provider cho tính năng search.

Chỉ cần đổi `SYLVIA_SEARCH_PROVIDER_TYPE` thành:

- `brave`
- `google`
- `none`

mà không cần thay đổi code.

---

## Demo

### 1) Trả lời roleplay

Video: [Watch demo](demo/traloiroleplay.mp4)

### 2) Tra cứu thông tin online

Video: [Watch demo](demo/tracuuthongtinonline.mp4)

### 3) Xử lý hình ảnh và context hội thoại

Video: [Watch demo](demo/xulyhinhvacontexttrochuyen.mp4)
