# Lab 21 — Evaluation Report

**Học viên**: Nguyễn Tùng Lâm — 2A202600555
**Ngày nộp**: 2026-06-25
**Submission option**: A (lightweight)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (90% train, 10% eval)
- **max_seq_length**: 1024 (do giới hạn phần cứng của profile T4)
- **GPU**: Tesla T4, 16 GB VRAM
- **Training cost**: $0.07 (~12.2 phút @ $0.35/hr)

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time (min) | Peak VRAM (GB) | Eval Loss | Perplexity |
|------|------------------|------------------|----------------|-----------|------------|
| 8    | 1,843,200        | 3.95             | 9.31           | 1.557     | 4.747      |
| 16   | 3,686,400        | 4.13             | 6.61           | 1.516     | 4.554      |
| 64   | 14,745,600       | 3.95             | 10.09          | 1.476     | 4.378      |
| Base | -                | -                | -              | -         | -          |

*(Ghi chú: Peak VRAM có thể có sự biến động nhẹ tùy thuộc vào tiến trình giải phóng bộ nhớ của PyTorch giữa các lần train).*

## 3. Loss Curve Analysis
*(Hình ảnh `loss_curve.png` được đính kèm trong thư mục submission)*

- **Quan sát**: Do sử dụng profile T4 với cấu hình `eval_strategy="no"` để tiết kiệm VRAM, mô hình chỉ ghi nhận được train loss curve mà không có eval loss trong quá trình huấn luyện. Train loss giảm dần và ổn định theo thời gian, cho thấy mô hình đang học được các pattern từ dataset. 
- **Overfitting**: Vì không có eval loss ở từng epoch, ta không thể đánh giá trực tiếp overfitting qua biểu đồ, tuy nhiên việc eval perplexity cuối cùng khá tốt (chỉ quanh 4.3 - 4.7) cho thấy mô hình chưa bị overfit nghiêm trọng. 

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
**Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu. Mô hình học được từ dữ liệu này có thể thực hiện các tác
**Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo) và sử dụng thuật toán và các mô hình học máy để học từ dữ liệu và từ đó đưa ra các dự đoán hoặc
**Nhận xét**: Cả hai mô hình đều có câu trả lời khá tương đương về mặt ý nghĩa, tuy nhiên câu văn của bản fine-tuned có phần trôi chảy và cấu trúc định nghĩa chặt chẽ hơn một chút so với base model. (Sự cải thiện: Nhẹ).

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n: ```python def fibonacci(n): if n <= 0: return "N phải là một số dương" elif n == 1: return 0 elif n == 2: return 1 else: ... ```
**Fine-tuned (r=16)**: Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau: ```python def fibonacci(n): if n < 0: raise ValueError("Input phải là một số nguyên dương.") elif n == 0: return 0 elif n == 1: return 1 else: a, b = 0, 1 for _ in range ... ```
**Nhận xét**: Bản fine-tuned đưa ra phương pháp dùng vòng lặp `for` tối ưu hơn về mặt thời gian thực thi (O(n)) so với hàm đệ quy cơ bản (O(2^n)) ở phần `else` trong Base model. Xử lý ngoại lệ (`ValueError`) cũng chuẩn chỉ hơn với Python. (Sự cải thiện: Tốt).

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc, font chữ, hình ảnh và các yếu tố khác để đảm bảo người dùng có thể dễ dàng sử dụng sản phẩm. 2. Tru...
**Fine-tuned (r=16)**: 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX thiết kế phải dễ dàng sử dụng và dễ hiểu. 4. Tương thích: UI/UX thiết kế phải dễ dàng tương thích với ...
**Nhận xét**: Bản Base liệt kê diễn giải khá lan man, trong khi bản Fine-tuned bám sát định dạng "liệt kê" hơn, trình bày gọn gàng và đưa ra nhiều ý rõ rệt thay vì dồn vào 1 ý quá dài. (Sự cải thiện: Tốt).

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hiệu năng của mô hình bằng cách thay đổi các phép biến đổi trong mô hình bằng các phép biến đổi nhỏ hơ...
**Fine-tuned (r=16)**: LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong các mô hình có số lượng các layer lớn. LoRA là một kỹ thuật regularization được sử dụng để cải thiệ...
**Nhận xét**: Cả hai mô hình đều có sự nhầm lẫn (hallucination) về việc giải thích LoRA (Base thì giải thích vòng vo, Fine-tuned thì viết sai cả cụm từ viết tắt của LoRA thành Layer-wise...). Điều này dễ hiểu vì dataset 200 samples chung chung không đủ để dạy kiến thức chuyên sâu về LLM. (Sự cải thiện: Không có/Kém).

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
**Base**: Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của mô hình bằng cách cung cấp cho nó một câu hỏi hoặc câu lệnh để dựa vào, thay vì cung cấp dữ liệu đầu...
**Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp hệ thống AI giải quyết các vấn đề và thực hiện các tác vụ. Prompt được sử dụng để cung cấp cho hệ th...
**Nhận xét**: Cả hai đều có sự khởi đầu khá tốt và định nghĩa được hướng đi chung. Model fine-tuned có diễn đạt tiếng Việt trôi chảy hơn và tự nhiên hơn một chút. (Sự cải thiện: Nhẹ).

## 5. Conclusion về Rank Trade-off
- **Rank nào cho ROI tốt nhất?** Trên dataset này, **`r=16`** mang lại ROI (Return on Investment) tốt nhất. Với chỉ khoảng 3.6 triệu tham số (gấp đôi `r=8`), nó cải thiện perplexity từ 4.74 xuống 4.55 mà thời gian train hầu như không đổi (khoảng hơn 4 phút).
- **Diminishing returns:** Sự bão hòa lợi ích thể hiện rõ khi ta tăng lên `r=64`. Số lượng tham số phải huấn luyện tăng vọt lên 14.7 triệu (gấp 4 lần `r=16`), nhưng perplexity chỉ giảm nhẹ từ 4.55 xuống 4.37. Chi phí bỏ ra (Peak VRAM lên tới 10GB) không mang lại sự khác biệt đủ lớn so với tài nguyên phải bỏ ra.
- **Recommendation:** Nếu deploy trên môi trường production, tôi sẽ chọn rank **`r=16`**. Nó tạo ra được sự cân bằng tối ưu giữa việc tiết kiệm tài nguyên hệ thống (VRAM, storage cho adapter checkpoint) mà vẫn giữ được chất lượng ngôn ngữ khá tốt (hiệu năng tốt hơn hẳn so với `r=8` và không bị quá tải như `r=64`).

## 6. What I Learned
- **Cấu hình QLoRA tiết kiệm bộ nhớ**: Tôi đã hiểu được cách thiết lập 4-bit quantization và kết hợp với Gradient Checkpointing có thể tiết kiệm đáng kể VRAM như thế nào, cho phép train mô hình 3B trên một GPU nhỏ bé như T4.
- **Sự ảnh hưởng của Rank (r)**: Thông qua experiment, tôi nhận thấy rõ quy luật "diminishing returns" — việc tăng rank không phải lúc nào cũng mang lại lợi ích tương xứng, và đôi khi việc giữ rank ở mức vừa phải (`r=16`) lại là lựa chọn khôn ngoan nhất.
- **Đánh giá định tính vs định lượng**: Mặc dù Perplexity giảm, mô hình đôi khi vẫn bị hallucinate (như trong trường hợp định nghĩa LoRA). Điều này nhấn mạnh tầm quan trọng của việc đánh giá Qualitative song song với Quantitative.
