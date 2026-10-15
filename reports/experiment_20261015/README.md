# 实验报告文件

- `实验报告.md`、`实验报告.pdf`：已完成的四事件单seed实验报告。
- `assets/`：指标图与训练曲线，不含原始卫星影像。
- `verification.json`：报告生成时对正式结果的配对核对。
- `build_report.py`、`build_pdf.py`、`report_sections.json`：作者当时使用的报告生成源码和中间文本，供审阅。它们使用本地审计文件、Windows中文字体，以及额外的 python-docx、reportlab、Pillow 库；不属于队友免重训复核的必需依赖。原始审计文件不在Release里，因此不保证仅凭复现ZIP可重新生成报告。请直接阅读已提供的MD/PDF。

模型、数据与结果的复核统一按[复核入口](../../REPRODUCE_AND_DIAGNOSE.md)操作。
