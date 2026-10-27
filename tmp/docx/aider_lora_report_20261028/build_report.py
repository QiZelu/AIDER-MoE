from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path('/Users/zelu/Desktop/Codes/AIDER-main')
EMBED_ROOT = Path('/Users/zelu/Desktop/Codes/AIGVQA Task Embedding')
WORK = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'docs'
OUT.mkdir(parents=True, exist_ok=True)
TARGET = OUT / 'AIDER_LoRA_CLIP_Adapter_工作记录_20261028.docx'

METRICS = {
    'static': {'count': 115, 'plcc': 0.9220314831962539, 'srocc': 0.9182698748010542, 'krocc': 0.7511741442711647, 'rmse': 0.04865805432200432},
    'temporal': {'count': 115, 'plcc': 0.9469592478126134, 'srocc': 0.9392135119145908, 'krocc': 0.776975072031779, 'rmse': 0.048320114612579346},
    'dynamic': {'count': 115, 'plcc': 0.6634988055925007, 'srocc': 0.689221352342052, 'krocc': 0.5045838238209366, 'rmse': 0.0698862075805664},
    'text_video': {'count': 115, 'plcc': 0.769905398184385, 'srocc': 0.7454264335890819, 'krocc': 0.5695517822042757, 'rmse': 0.07039207220077515},
    'macro': {'plcc': 0.8255987336964383, 'srocc': 0.8230327931616948, 'krocc': 0.650571205582039, 'rmse': 0.059314112178981304},
}
for key in ('plcc', 'srocc', 'krocc', 'rmse'):
    value = sum(METRICS[task][key] for task in ('static', 'temporal', 'dynamic', 'text_video')) / 4
    assert abs(value - METRICS['macro'][key]) < 1e-12, (key, value)

comparison_path = EMBED_ROOT / 'results/criterion7-intent-projection-v1/comparison_summary.csv'
with comparison_path.open() as handle:
    CLIP_ROWS = list(csv.DictReader(handle))
manifest_path = EMBED_ROOT / 'data/criterion7_intent_v1/manifest.json'
manifest = json.loads(manifest_path.read_text())
assert manifest['total_descriptions'] == 5376
assert [manifest['stats'][s]['descriptions'] for s in ('train', 'validation', 'test')] == [4032, 672, 672]

evidence = {
    'record_date': '2026-10-28',
    'aider_lora': {'variant': 'two_stage', 'phase_epochs': [5, 5], 'checkpoint': 'best',
                   'identity_confirmed_by_user': True, 'metrics': METRICS,
                   'source': 'User supplied metrics and checkpoint identity in this conversation'},
    'clip_projection_version': 'criterion7-intent-projection-v1',
    'clip_comparison_csv': str(comparison_path),
    'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (
        ROOT / 'AIDER-LoRA/README.md', ROOT / 'AIDER-LoRA/config.yaml',
        ROOT / 'AIDER-LoRA/models/fusion.py', ROOT / 'AIDER-LoRA/models/lora.py',
        ROOT / 'AIDER-LoRA/models/film.py', comparison_path, manifest_path,
        EMBED_ROOT / 'work/intent_v1/train_projection.py',
    )},
}
(WORK / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin, sec.bottom_margin = Inches(.64), Inches(.64)
sec.left_margin, sec.right_margin = Inches(.75), Inches(.75)
sec.footer_distance = Inches(.27)


def font_style(style, size, bold=False):
    style.font.name = 'Arial'
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = RGBColor(0, 0, 0)
    rf = style.element.get_or_add_rPr().get_or_add_rFonts()
    rf.set(qn('w:ascii'), 'Arial')
    rf.set(qn('w:hAnsi'), 'Arial')
    rf.set(qn('w:eastAsia'), 'Noto Sans CJK SC')
    for key in ('w:asciiTheme', 'w:hAnsiTheme', 'w:eastAsiaTheme', 'w:cstheme'):
        rf.attrib.pop(qn(key), None)
    ppr = style.element.find(qn('w:pPr'))
    if ppr is not None:
        borders = ppr.find(qn('w:pBdr'))
        if borders is not None:
            ppr.remove(borders)


font_style(doc.styles['Normal'], 11)
normal = doc.styles['Normal'].paragraph_format
normal.line_spacing = Pt(16)
normal.space_after = Pt(7)
normal.widow_control = True
snap = OxmlElement('w:snapToGrid')
snap.set(qn('w:val'), '0')
doc.styles['Normal'].element.get_or_add_pPr().append(snap)
font_style(doc.styles['Title'], 22, True)
doc.styles['Title'].paragraph_format.space_after = Pt(9)
doc.styles['Title'].paragraph_format.line_spacing = Pt(30)
font_style(doc.styles['Subtitle'], 10.5)
doc.styles['Subtitle'].paragraph_format.space_after = Pt(12)
doc.styles['Subtitle'].paragraph_format.line_spacing = Pt(15)
for name, size in (('Heading 1', 15), ('Heading 2', 12)):
    font_style(doc.styles[name], size, True)
    p = doc.styles[name].paragraph_format
    p.space_before, p.space_after = Pt(11 if name == 'Heading 1' else 8), Pt(6)
    p.keep_with_next = True
    p.keep_together = True
    p.line_spacing = Pt(22 if name == 'Heading 1' else 17)
for name in ('Caption', 'Source Note'):
    if name not in doc.styles:
        doc.styles.add_style(name, 1)
    font_style(doc.styles[name], 9.5)
    doc.styles[name].paragraph_format.space_after = Pt(5)
    doc.styles[name].paragraph_format.line_spacing = Pt(13.5)
font_style(doc.styles['Source Note'], 10)
font_style(doc.styles['Footer'], 9)

footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
footer.paragraph_format.space_after = Pt(0)
run = footer.add_run('第 ')
run.font.size = Pt(9)
field = OxmlElement('w:fldSimple')
field.set(qn('w:instr'), 'PAGE')
footer._p.append(field)
footer.add_run(' 页').font.size = Pt(9)


def para(text, style=None, bold_lead=None):
    p = doc.add_paragraph(style=style)
    if bold_lead and text.startswith(bold_lead):
        p.add_run(bold_lead).bold = True
        p.add_run(text[len(bold_lead):])
    else:
        p.add_run(text)
    return p


def heading(text, level=1):
    return doc.add_paragraph(text, style=f'Heading {level}')


def page(title):
    doc.add_page_break()
    heading(title)


def table(headers, rows, widths, numeric=(), size=10, emphasize_last=False):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for col, width in zip(t.columns, widths):
        col.width = Inches(width)
    props = t._tbl.tblPr
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        line = OxmlElement(f'w:{edge}')
        line.set(qn('w:val'), 'single')
        line.set(qn('w:sz'), '4')
        line.set(qn('w:color'), 'D9D9D9')
        borders.append(line)
    props.append(borders)
    margins = OxmlElement('w:tblCellMar')
    for side, value in (('top', '90'), ('bottom', '90'), ('left', '105'), ('right', '105')):
        node = OxmlElement(f'w:{side}')
        node.set(qn('w:w'), value)
        node.set(qn('w:type'), 'dxa')
        margins.append(node)
    props.append(margins)
    header_row = t.rows[0]
    repeat = OxmlElement('w:tblHeader')
    header_row._tr.get_or_add_trPr().append(repeat)
    all_rows = [list(headers)] + [list(row) for row in rows]
    for row_index, values in enumerate(all_rows):
        row = header_row if row_index == 0 else t.add_row()
        cant_split = OxmlElement('w:cantSplit')
        row._tr.get_or_add_trPr().append(cant_split)
        for j, (cell, value, width) in enumerate(zip(row.cells, values, widths)):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.line_spacing = Pt(13)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j in numeric or row_index == 0 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(str(value))
            r.font.size = Pt(size)
            r.font.bold = row_index == 0 or (emphasize_last and row_index == len(all_rows)-1)
            r.font.color.rgb = RGBColor(255, 255, 255) if row_index == 0 else RGBColor(0, 0, 0)
            shading = OxmlElement('w:shd')
            shading.set(qn('w:fill'), '213B56' if row_index == 0 else ('F2F5F9' if row_index % 2 else 'FFFFFF'))
            cell._tc.get_or_add_tcPr().append(shading)
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(1)
    after.paragraph_format.space_before = Pt(0)
    after.paragraph_format.line_spacing = 1
    after.add_run().font.size = Pt(1)
    return t


def mr(text):
    r = OxmlElement('m:r')
    node = OxmlElement('m:t')
    node.text = text
    r.append(node)
    return r


def sub(base, index):
    node = OxmlElement('m:sSub')
    e, s = OxmlElement('m:e'), OxmlElement('m:sub')
    e.append(mr(base)); s.append(mr(index))
    node.extend([e, s])
    return node


def frac(num, den):
    node = OxmlElement('m:f')
    n, d = OxmlElement('m:num'), OxmlElement('m:den')
    n.append(mr(num)); d.append(mr(den))
    node.extend([n, d])
    return node


def equation(parts):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(7)
    p.paragraph_format.line_spacing = Pt(32)
    math_para, math = OxmlElement('m:oMathPara'), OxmlElement('m:oMath')
    for element in parts:
        math.append(mr(element) if isinstance(element, str) else element)
    math_para.append(math)
    p._p.append(math_para)
    return p


# Page 1
para('AIDER LoRA 与 CLIP Adapter 工作记录', 'Title')
para('研究进展与实验结果    2026年10月28日', 'Subtitle')
para('目前完成了两项相互关联的工作：AIDER-LoRA 使用共享 Transformer 与任务专属低秩更新完成多维视频评价；CLIP 意图 Adapter 使用评估准则文本学习更符合评价意图的表示。前者已获得视频评分结果，后者已在文本意图关系和类别检索上获得改善。两者组合后的下游增益尚需单独验证。')
heading('一 AIDER LoRA 当前效果')
para('本次结果对应两阶段 5+5 训练的 best 检查点。四个任务分别在 115 个视频上评价，共 460 条视频与任务评分记录。表中指标保留四位小数，原始数值用于宏平均核对。[S1]')
metric_names = [('static', 'static 静态质量'), ('temporal', 'temporal 时序质量'), ('dynamic', 'dynamic 动态程度'), ('text_video', 'text_video 文本一致性'), ('macro', '宏平均')]
table(['任务', '样本', 'PLCC ↑', 'SROCC ↑', 'KROCC ↑', 'RMSE ↓'],
      [[label, METRICS[key].get('count', '-'), *[f'{METRICS[key][m]:.4f}' for m in ('plcc','srocc','krocc','rmse')]] for key,label in metric_names],
      [1.7,.55,1.19,1.19,1.19,1.18], numeric=(1,2,3,4,5), emphasize_last=True, size=10)
para('宏平均 PLCC 为 0.8256，SROCC 为 0.8230。static 与 temporal 的排序相关性较高，SROCC 分别为 0.9183 与 0.9392；dynamic 与 text_video 仍有改善空间，分别为 0.6892 与 0.7454。', bold_lead='宏平均 PLCC 为 0.8256，SROCC 为 0.8230。')
para('PLCC 衡量线性相关性；SROCC 衡量秩相关性，也常记为 SRCC；KROCC 衡量排序一致性；RMSE 衡量当前标签尺度下的预测误差。宏指标为四个任务指标的算术平均，其中宏 RMSE 不是合并所有误差后重新计算的 RMSE。', 'Source Note')
heading('目前的性能观察', 2)
para('研究者观察到 LoRA 组相较 MoE 的综合能力提升，Text-Video 的改善尤其明显，并已接近 AIGV-Assessor，部分维度超过。当前绝对分数见上表；模型间的提升幅度和持平结论，需要结合相同划分、相同选模规则下的逐维对照结果解释。')
heading('CLIP 意图 Adapter 的结果概览', 2)
para('在 criterion7-intent-projection-v1 的文本评估中，一级类别宏召回率为 82.16% ± 0.98%，二级维度宏召回率为 72.57% ± 3.05%，均高于原始 CLIP。细分意图排序存在取舍，完整对照见第四节。这些数值属于文本表示评估，不是视频质量预测指标。[S5]')

# Page 2
page('二 AIDER LoRA 做了什么')
para('保留原 AIDER 的三路视频证据、伪影先验和跨分支 FiLM，将融合层中的完整 FFN 专家池替换为普通共享 Transformer，并在 Attention 与 FFN 的线性投影中加入任务专属 LoRA。[S2]')
table(['部分', '原 AIDER MoE 参考实现', 'AIDER LoRA'], [
    ['视频证据', 'VITAL 技术特征与伪影先验；SlowFast 运动特征；BLIP 文本视频特征', '保留三路、伪影先验与 Artifact Query'],
    ['特征调制', '任务私有 FiLM 与 alpha，其他两路视频特征提供动态残差', '保留；采用 video_only 条件'],
    ['时序融合', '2 层，宽度 1536，12 个注意力头', '相同；输入仍为 8 个时间 token'],
    ['Attention', 'Q K V 与输出投影的参数共享', '四个投影均加入任务专属 LoRA'],
    ['FFN', '每层 7 个路由专家选 2 个，另加 1 个常开共享专家', '每层 1 个共享 FFN，fc1 与 fc2 均加入任务专属 LoRA'],
    ['任务分流', '任务向量决定专家与混合权重', '显式任务 ID 选择对应 LoRA；回归头仍为任务私有'],
], [1.02,2.99,2.99], size=9.5)
heading('任务专属低秩更新', 2)
equation([sub('W','t'), ' = ', sub('W','shared'), ' + ', frac('16','8'), sub('B','t'), sub('A','t')])
para('每个任务在每层拥有 Q、K、V、输出投影、fc1、fc2 六处独立低秩更新。rank 为 8，alpha 为 16，LoRA dropout 为 0.05；A 独立随机初始化，B 初始化为零。初始更新为零，推理时可写为上式。混合批次只执行出现任务的 Adapter。')
heading('融合模块的参数规模', 2)
table(['参数范围', '参数量'], [
    ['原 MoE 融合模块 含四任务私有 Gate', '94,534,712'],
    ['LoRA 版共享 Transformer 基础参数', '28,345,344'],
    ['四任务全部 LoRA 每任务 294,912', '1,179,648'],
    ['LoRA 融合模块合计', '29,524,992'],
], [5.05,1.95], numeric=(1,), size=10, emphasize_last=True)
para('按当前源码静态计算，融合参数量下降约 68.8%。此口径排除三路编码器、FiLM 和回归头；原 MoE 采用当前根目录的私有 Gate 实现。该比例不代表整模型参数或峰值显存的降幅。', 'Source Note')
heading('5 加 5 两阶段训练', 2)
para('第 1 至 5 轮训练共享 Transformer，旁路并冻结 LoRA；第 6 至 10 轮冻结融合基础线性层、偏置与 LayerNorm，启用任务 LoRA。FiLM、alpha、私有 Head、BLIP 文本与交叉注意力及其他原可训练模块继续训练。切换使用第 5 轮 last，保留其他模块 Adam 状态与全局学习率进度。[S3]')

# Page 3
page('三 CLIP 意图 Adapter 做了什么')
para('该工作针对评价描述的表示问题：相近的视觉语义和共同措辞可能使不同评价意图在原始 CLIP 空间中距离较近。通过专门构建的准则文本与关系监督，训练 CLIP 后的投影，使表示更符合评价意图的同义、差异与层级关系。此处记录 criterion7-intent-projection-v1，其他历史 Adapter 版本不混用结果。[S4]')
heading('评估意图描述数据集', 2)
para('criterion7_intent_v1 采用四个一级类别和七个二级维度，每个二级维度 768 条英文描述，总计 5,376 条。描述由 56 个具体意图、448 条措辞和 36 个模板构成，因而描述条数不能视为独立意图数量。[S4]')
table(['一级类别', '二级评价维度'], [
    ['质量评价 Q', '空间质量 SQ；动态质量 DQ；美学质量 AQ'],
    ['真实感 R', '物理常识 PC；解剖合理性 AN'],
    ['内容丰富度 C', '动态程度 DD'],
    ['文本视频对齐 A', '文本对齐 TA'],
], [2.0,5.0], size=10)
table(['划分', '描述数量', '具体意图数量', '每个二级维度'], [
    ['训练集', '4,032', '42', '576 条'],
    ['验证集', '672', '7', '96 条'],
    ['测试集', '672', '7', '96 条'],
], [1.6,1.8,1.8,1.8], numeric=(1,2,3), size=10)
para('划分同时留出具体准则与模板族。正样本严格保留同一个具体评估意图并改变措辞；负样本改变二级维度，同一级类别内为较弱负例，跨一级类别为较强负例。同二级类别但具体准则不同的描述不直接作为正样本或负样本。')
heading('模型与训练目标', 2)
para('冻结 CLIP ViT-B/32 文本编码器，输入 512 维单位化特征；接 Linear 512 到 512、ReLU、Linear 512 到 512，再作 L2 归一化。仅训练投影的 525,312 个参数，无分类头或 Root 表。[S4]')
para('损失同时优化四项关系：同意图正样本对齐、同一级类别内的区分、跨一级类别的区分，以及同级负例应比跨级负例更接近的层级排序。相应权重依次为 0.20、0.50、1.25、0.50；同级、跨级和层级 margin 分别为 0.10、0.35、0.15。')
heading('训练协议', 2)
para('使用 seed 42、43、44，AdamW 学习率 0.00001、weight decay 0.01，训练 1,000 个全批次更新步，梯度范数裁剪为 1。每个训练锚点有 2 个正例、6 个负例；按一级类别及其子类别平衡锚点贡献。每 10 步评价验证损失，选择最小验证复合损失的检查点；所有检查点固定后再评估测试集。[S4]')

# Page 4
page('四 CLIP 意图 Adapter 的效果')
para('以下结果使用相同测试关系。两个投影版本报告 seed 42、43、44 的均值与样本标准差；原始 CLIP 为固定编码器结果。此前层级投影指 criterion7-hierarchy-projection-v4 的 gentle 配置。[S5]')
table(['指标', '原始 CLIP', '此前层级投影', '意图投影'], [
    [row['metric'], row['raw_clip'], row['previous_mean_sample_std'], row['intent_mean_sample_std']] for row in CLIP_ROWS
], [2.8,1.05,1.575,1.575], numeric=(1,2,3), size=9.5)
para('p 表示保留具体意图的正样本，s 表示同一级类别内其他二级维度的负样本，c 表示跨一级类别负样本；关系是否成立由余弦相似度比较。类别召回率来自训练集类中心检索诊断，未训练分类头。“±”为三个种子的样本标准差，不是置信区间。', 'Source Note')
heading('已观察到的改善', 2)
para('相较原始 CLIP，一级类别宏召回率提高 10.89 个百分点，二级维度宏召回率提高 5.46 个百分点。正样本优于跨一级负例的比例从 24.23% 提高到 79.63%，说明投影更符合当前意图关系定义。')
para('相较此前层级投影，一级与二级宏召回率分别提高 5.80 和 14.19 个百分点；同级负例优于跨级负例的比例提高约 14.33 个百分点，完整层级排序提高约 8.11 个百分点。差值按原始结果计算，四舍五入后可能与表内百分数直接相减略有差异。')
heading('仍然存在的取舍', 2)
para('同意图正样本优于同一级负例的比例，从此前版本的 61.34% 降到 56.45%，下降 4.89 个百分点。因此，当前投影改善了跨大类区分和层级结构，但细粒度意图区分没有同步全面改善。')
para('数据为合成准则和模板组合；验证集与测试集各只有 7 个独立具体意图，每个二级维度一个。672 条测试句子不能等同于 672 个独立评价意图。现有结果支持在该意图体系上的文本表示改善，对自然表述、跨数据集准则及视频 MOS 的作用需分别评价。')
para('意图投影版本同时改变了描述覆盖与关系构造，相较此前版本的差异不能全部归因于某一个损失项。Rubric、反事实场景、宽泛名称和边界案例保留作辅助诊断，没有额外作为本轮损失输入。[S4]')

# Page 5
page('五 两项工作的关系与当前判断')
para('AIDER-LoRA 的任务能力目前主要由任务 ID 和私有参数承担。LoRA 按 ID 选择，不读取 CLIP 向量；FiLM 的任务基础线性层与 alpha 也为任务私有。当前 video_only 动态残差读取其他分支的视频摘要，不把任务向量作为其输入。[S2]')
heading('意图向量在私有 FiLM 中的作用', 2)
equation(['[', sub('γ','t'), ', ', sub('β','t'), '] = ', sub('W','t'), sub('e','t'), ' + ', sub('b','t'), ' = ', sub('c','t')])
para('每个固定任务只有一个固定描述向量，因此该任务的基础调制输出可以直接表示为可学习常量。即使所有任务输入同一个向量，各自独立的线性层仍能学到不同的调制。alpha 同样可以表示为每任务一个可学习标量经 sigmoid 后的值。')
para('这说明意图 Adapter 对当前四任务模型在表达能力上并非必需。它可能改变训练优化、参数正则化与归纳偏置，实际收益仍需视频评价消融证明。文本向量之间的余弦距离变大，本身不能保证下游预测提高。')
heading('更适合意图表示的使用方式', 2)
para('若需要自然语言选择已有评价维度，意图模块可以先将用户描述映射为任务 ID，再调用对应 LoRA。若需要意图表示直接影响融合，则可由共享 FiLM 生成器、共享 Query 生成器，或共享超网络根据意图生成适配参数。此时需要检验同义改写、细则变化和留出任务上的行为。上述连接方式属于后续设计，尚未包含在本次结果中。')
heading('优先完成的对照', 2)
table(['对照', '需要回答的问题'], [
    ['固定同一向量 随机向量 原始 CLIP 意图投影', '统一维度和范数，检验文本意图几何是否改善当前私有模块的结果'],
    ['普通共享 Transformer 与加入任务 LoRA', '区分移除专家池和增加低秩任务适配各自的收益'],
    ['仅 FFN LoRA 与 Attention 加 FFN LoRA', '检验 Text-Video 是否主要受益于 Attention 投影适配'],
    ['joint 与两阶段 5 加 5', '区分共享基础参数持续训练和分阶段冻结的影响'],
], [3.45,3.55], size=9.5)
para('当前意图投影基于 ViT-B/32 的 512 维特征；AIDER-LoRA 的固定任务表来自 ViT-L/14 的 768 维投影。接入实验应匹配底层编码器和维度，避免将更换 CLIP 的影响归入 Adapter。')
para('另一个可检验的简化是直接学习任务基础 gamma、beta 与 alpha。按当前四任务三分支配置，这部分参数可由 9,452,548 减少至 12,292，不含共享视频动态调制、LoRA 与 Head。固定任务下可表达相同的任务常量，但优化动态改变，需要重新训练验证；该简化尚未实施。')

# Page 6
page('六 实验口径与记录依据')
heading('AIDER LoRA 实验设置', 2)
table(['项目', '当前记录'], [
    ['实验与检查点', 'two_stage；前 5 轮 dense，后 5 轮 adapter；best 检查点；模型 seed 42'],
    ['数据规模', '训练 461 个视频，共 1,844 条任务记录；测试 115 个视频，共 460 条任务记录'],
    ['训练预算', 'batch 8，尾批 4 条；每轮 231 步，共 10 轮、2,310 步'],
    ['优化设置', 'AdamW；基础学习率 0.0001；BLIP 文本部分 0.00001；weight decay 0.05'],
    ['精度与调度', 'BF16 前向；FP32 可训练参数与 Adam 状态；warmup 116 步，连续余弦调度'],
    ['损失', 'FP32 Smooth L1，beta 0.1，样本均值；不包含 MoE 互信息或其他辅助损失'],
    ['选模口径', '第 6 至 10 轮按测试集四任务宏平均 SROCC 严格提升选择 best；并列保留较早轮次'],
], [1.35,5.65], size=9.5)
para('测试集参与了检查点选择，因此该 best 分数包含选模影响。正式比较泛化能力时，应统一基线的数据划分与选模协议，并重复关键模型种子或划分。当前代码归档保存源码与配置，完整运行还依赖原服务器的数据、预训练权重和缓存路径。[S3]')
heading('记录依据', 2)
sources = [
    ('S1', 'AIDER-LoRA 指标快照', '2026年10月28日研究者提供原始 metrics，并确认对应两阶段 5+5 的 best。表格保留四位小数；宏平均已按四维算术平均核对。'),
    ('S2', 'AIDER-LoRA 模型源码', '/Users/zelu/Desktop/Codes/AIDER-main/AIDER-LoRA/models/ 下的 aider.py、fusion.py、lora.py、film.py、task_private.py；MoE 参数对照为父项目 models/。'),
    ('S3', 'AIDER-LoRA 训练与归档', '/Users/zelu/Desktop/Codes/AIDER-main/AIDER-LoRA/ 下的 README.md、config.yaml、COPY_INFO.md、engine/optim.py 与 engine/trainer.py。'),
    ('S4', '意图投影数据与训练', '/Users/zelu/Desktop/Codes/AIGVQA Task Embedding/ 下的 data/criterion7_intent_v1/manifest.json、work/intent_v1/train_projection.py 与 aigvqa_mapper/criterion_projection.py。'),
    ('S5', '意图投影效果', '/Users/zelu/Desktop/Codes/AIGVQA Task Embedding/results/criterion7-intent-projection-v1/ 下的 comparison_summary.csv、report.json 与 protocol.json。'),
]
for code, title, text in sources:
    p = para(f'[{code}] {title}。{text}', 'Source Note')
    p.paragraph_format.space_after = Pt(6)

doc.core_properties.title = 'AIDER LoRA 与 CLIP Adapter 工作记录'
doc.core_properties.subject = '两阶段任务 LoRA 视频评价与 CLIP 评估意图投影的实现和实验记录'
doc.core_properties.author = ''
doc.core_properties.keywords = 'AIDER-LoRA CLIP Adapter AIGVQA 多任务学习 意图表示'
doc.save(TARGET)
print(json.dumps({'docx': str(TARGET), 'evidence': str(WORK/'evidence.json'),
                  'aider_macro': METRICS['macro'], 'clip_comparison_rows': len(CLIP_ROWS)}, ensure_ascii=False))
