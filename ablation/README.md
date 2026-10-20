# Kết quả ablation reverse noise

Thí nghiệm trên checkpoint EMA 200.000 updates, T=1000. λ nhân độ lệch chuẩn
của nhiễu trong reverse sampling; các điều kiện đối chiếu dùng chung bank x_T
và reverse seeds.

- [Báo cáo tiếng Việt](reverse_noise/ema200k/report_vi.md): phương pháp, kết quả,
  kiểm chứng và chi phí.
- [Bảng FID/IS](reverse_noise/ema200k/metrics.csv): λ=1; 0,5; 0,1; 0,
  mỗi mức 5.000 ảnh.
- [Khảo sát trực quan quanh λ=1](reverse_noise/ema200k/qualitative_128/README.md):
  λ=0,75; 0,9; 1,1; 1,25, mỗi mức 128 ảnh, ghép theo ID với baseline λ=1.
- [Hướng dẫn mã sampling](../ABLATION.md).

Repo lưu báo cáo, CSV và toàn bộ grid/trang ghép dùng trong báo cáo.
Các file `reproducibility.zip` chứa tensor, PNG riêng và metadata được giữ local,
loại khỏi Git để tránh upload dữ liệu lớn.
