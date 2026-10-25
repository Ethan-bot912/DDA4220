"""Prepare reviewable validation-only results; never train, infer, or invoke Git."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/within_event_mm/original_baseline'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {}
    runs = {}
    for regime, folder in [('cross_event', ROOT / 'reports/expanded_seed42'), ('within_event', OUT)]:
        runs[regime] = {}
        for model in ['b1', 'b2']:
            run_folder = folder / model if regime == 'cross_event' else folder / model / 'seed42/attempt01'
            source = run_folder / 'summary.json'
            s = read(source)
            assert s['completed_epochs'] == 20 and not s['test_evaluated']
            assert len(s['history']) == 20
            best = max(s['history'], key=lambda h: h['val']['macro_f1'])
            assert best['epoch'] == s['best_epoch']
            assert best['val'] == s['best_val'] and best['per_event'] == s['best_per_event']
            if regime == 'within_event':
                metadata = read(run_folder / 'run_metadata.json')
                for key in ('config', 'mode', 'manifest_sha256', 'code_sha256', 'versions', 'git_commit'):
                    assert metadata[key] == s[key], (model, key)
                history = [json.loads(line) for line in (run_folder / 'history.jsonl').read_text(encoding='utf-8').splitlines()]
                assert history == s['history'], model
            for metrics in [s['best_val'], *s['best_per_event'].values()]:
                cm = metrics['confusion']
                assert sum(map(sum, cm)) == metrics['n']
                f1s = []
                for i, (label, values) in enumerate(metrics['per_class'].items()):
                    assert sum(cm[i]) == values['support']
                    denom = sum(cm[i]) + sum(row[i] for row in cm)
                    f1 = 2 * cm[i][i] / denom if denom else 0
                    assert abs(f1 - values['f1']) < 1e-12
                    f1s.append(f1)
                assert abs(sum(f1s) / 4 - metrics['macro_f1']) < 1e-12
            runs[regime][model] = s
            sources[str(source.relative_to(ROOT)).replace('\\', '/')] = sha(source)
    within = runs['within_event']
    for key in ('config', 'manifest_sha256', 'code_sha256', 'versions', 'git_commit'):
        assert within['b1'][key] == within['b2'][key], key
    records = []
    classes = []
    for regime, models in runs.items():
        for model, s in models.items():
            for scope, m in [('pooled', s['best_val']), *s['best_per_event'].items()]:
                records.append(dict(regime=regime, model=model, scope=scope, n=m['n'], best_epoch=s['best_epoch'], accuracy=m['accuracy'], macro_f1=m['macro_f1'], balanced_accuracy=m['balanced_accuracy']))
                for label, values in m['per_class'].items():
                    classes.append(dict(regime=regime, model=model, scope=scope, label=label, **values))
    for name, rows in [('comparison.csv', records), ('per_class.csv', classes)]:
        with (OUT / name).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(OUT / 'comparison.json', {'scope': 'validation only; selected checkpoints; different training pools and validation samples', 'test_evaluated': False, 'overall_and_events': records, 'per_class': classes})
    sections = ['# 事件内与跨事件划分：seed42 验证结果分析',
        '本报告整理已完成的训练记录，不新增训练、推理或测试评估。两种划分下 B1/B2 均完成20轮。事件内合并验证 Macro F1 提高约20个百分点，主要改善来自 Matthew；这是诊断性证据，不是跨事件泛化提升的证明。',
        '## 1. 研究问题与实验过程',
        '原实验用5个事件训练，在未见过的 Matthew、Moore 上验证。本次只取这两个事件，按 (event, tile_id) 整块分配到约60%训练、20%验证、20%内部测试；同一块的建筑不可跨集合。确定性贪心与移动/交换精修平衡建筑数和类别数，划分被冻结，没有按成绩重抽。',
        '训练前完成环境修复、31项Python测试、10项串联检查、合成训练/续训验收、数据审计和数据锁核验（历史执行证据见本机 output/within_event_mm/freeze）。训练版本为 3ba5e61dd92f7400e1b169c1638a770e10820d03，B1先运行，B2随后运行并从断点完成；2026-10-24两者均完成20轮。报告发布版本不冒充训练版本。',
        'ResNet-18随机初始化、无预训练/增强、crop128/context16、batch32、AdamW lr0.001/weight_decay0.01、20轮cosine、seed42、float32均保持一致。类别权重各自只用训练集计算；各模型按合并验证Macro F1严格提升选best，所有分事件指标来自同一个best。',
        table(['设置', '跨事件 expanded', '事件内 within-event'], [
            ['训练数据', '5事件，104152栋', 'Matthew＋Moore，26976栋'],
            ['验证数据', 'Matthew＋Moore全体45002栋', '两事件9014栋，原验证池的子集'],
            ['内部测试', '无本次事件内测试划分', '9012栋，未评估'],
            ['防泄漏单位', '事件分离，tile也分离', 'tile分离，事件共享'],
            ['问题', '能否迁移到未见灾害事件', '能否识别已见事件的其他影像块']]),
        '当前只验证tile标识不重叠，不代表已经验证跨tile的地理距离、影像覆盖不重叠或空间独立性。项目级封存事件与这9012栋内部test是两套不同范围，均没有本次测试成绩。',
        '## 2. 总体与分事件结果']
    rows = []
    for scope in ['pooled', 'hurricane-matthew', 'moore-tornado']:
        for model in ['b1', 'b2']:
            a, b = [next(r for r in records if r['regime'] == regime and r['model'] == model and r['scope'] == scope) for regime in runs]
            rows.append([scope, model.upper(), a['n'], b['n'], f"{a['macro_f1']:.6f}", f"{b['macro_f1']:.6f}", f"{100*(b['macro_f1']-a['macro_f1']):+.2f}"])
    sections.append(table(['范围', '模型', '旧验证n', '新验证n', '跨事件F1', '事件内F1', '差值/百分点'], rows))
    sections.append(table(['模型', '指标', '跨事件', '事件内', '差值/百分点'], [[model.upper(), metric, f"{runs['cross_event'][model]['best_val'][metric]:.6f}", f"{runs['within_event'][model]['best_val'][metric]:.6f}", f"{100*(runs['within_event'][model]['best_val'][metric]-runs['cross_event'][model]['best_val'][metric]):+.2f}"] for model in ['b1','b2'] for metric in ['accuracy','balanced_accuracy']]))
    sections += ['## 3. 哪些类别改善，哪些困难仍在', '以下为合并验证集；分事件的每类precision/recall/F1及support完整保存在 per_class.csv。']
    rows = []
    for model in ['b1','b2']:
        for label, b in runs['within_event'][model]['best_val']['per_class'].items():
            a = runs['cross_event'][model]['best_val']['per_class'][label]
            rows.append([model.upper(), label, a['support'], b['support'], f"{a['f1']:.4f}", f"{b['f1']:.4f}", f"{100*(b['f1']-a['f1']):+.2f}", f"{b['precision']:.4f}/{b['recall']:.4f}"])
    sections.append(table(['模型','类别','旧n','新n','旧F1','新F1','差值/百分点','新P/R'], rows))
    sections += ['轻度损毁与完全摧毁的合并F1改善突出，但major仍较弱。B1在Matthew的no_damage召回率仅0.1515，major召回率0.2018；不能用总体提升掩盖残余错误。Moore的major验证样本仅90栋，minor仅178栋，类别分数容易波动。',
        '## 4. 灾前融合是否更有用',
        '跨事件合并 B2−B1 为−1.115个百分点；事件内为−1.565个百分点，整体仍未胜出。事件内Matthew上B2比B1高0.248个百分点、Moore高5.068个百分点。分事件都略高而合并更低并不矛盾：合并Macro F1基于合并混淆矩阵计算，不是事件分数的平均。不能据此宣称灾前信息普遍有效，也不能断言其完全无效。',
        '## 5. 如何解释划分的影响',
        '模型接触目标事件的训练样本后，Matthew分数明显恢复，支持事件分布差异是旧低分的重要因素之一。Matthew的minor占比54.5%、no_damage占比17.9%，与旧训练池明显不同，标签先验差异是可能机制；视觉环境、标注与空间相关性也可能参与，当前实验无法拆分其贡献。',
        '协议原文中“只差划分一个变量”的说法应理解为诊断目的，不能作为严格因果描述：训练事件、训练样本量、类别权重、每轮更新步数和验证样本均同时改变。旧验证池还包含如今用作训练的样本。这里报告的是两套评估条件的描述性差异，不是同一独立测试集上的配对提升或划分方法的净因果效应。协议保留原文以便溯源，本报告明确更正解释边界。',
        '仅有一个训练seed、一个确定性划分；未进行显著性检验或跨seed稳定性检验。验证集参与best选择，存在选优偏差。B1在Moore下降说明改善并不一致。内部test未评估，不能把当前验证数写成最终test结论。',
        '最早四事件实验只用Moore验证，B1/B2为0.601236/0.602278；它与扩大实验的合并验证范围不同，不能串成一个不加说明的性能排行榜。',
        '## 6. 后续工作（未执行）',
        '冻结所有选择后，经用户授权对内部test进行一次最终评估。若要更严格估计划分/目标事件暴露的影响，应事先定义固定且未使用过的评估tile、匹配训练规模与更新预算、多个训练seed及空间隔离方案。已有验证结果可以用于形成假设，不能回头挑划分以放大提升。',
        '## 7. 证据与复核',
        'b1/seed42/attempt01/、b2/seed42/attempt01/保存未经改写的summary、history和run_metadata（保留当时本机路径和git_dirty记录）。split_map与split_report为原始冻结副本。comparison表由原始summary导出。运行 `python tools/prepare_within_event_publication.py` 可直接使用仓库精选记录重建报告，无需数据清单或影像；脚本会核对20轮、best选择、混淆矩阵重算F1、记录内来源一致性，不运行模型。仅下载本报告目录时，可直接用保存的混淆矩阵复核指标。',
        '![划分对比](split_comparison.png)', '![训练过程](learning_curves.png)']
    (OUT / 'analysis_report.md').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax, scope in zip(axes, ['pooled','hurricane-matthew','moore-tornado']):
        for i, regime in enumerate(runs):
            vals=[next(r['macro_f1'] for r in records if r['regime']==regime and r['model']==m and r['scope']==scope) for m in ['b1','b2']]
            bars=ax.bar([j+(i-.5)*.35 for j in range(2)], vals, width=.35,label=regime)
            ax.bar_label(bars,fmt='%.3f',fontsize=8)
        ax.set(xticks=[0,1],xticklabels=['B1','B2'],ylim=(0,1),title=scope)
    axes[0].set_ylabel('Validation Macro F1'); axes[0].legend(fontsize=8)
    fig.suptitle('Different training pools and validation samples; seed 42; not test results')
    fig.tight_layout(); fig.savefig(OUT/'split_comparison.png',dpi=160); plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax, model in zip(axes,['b1','b2']):
        for regime in runs:
            s=runs[regime][model]
            ax.plot([h['epoch'] for h in s['history']],[h['val']['macro_f1'] for h in s['history']],label=regime)
        ax.set(title=model.upper(),xlabel='Epoch',ylabel='Pooled validation Macro F1',ylim=(0,1)); ax.legend()
    fig.tight_layout(); fig.savefig(OUT/'learning_curves.png',dpi=160); plt.close(fig)
    write_json(OUT/'publication_verification.json', {'validation_only':True, 'test_evaluated':False, 'training_commit':runs['within_event']['b1']['git_commit'], 'checks':['20 epochs each','selected best matches history','confusion counts and class F1 recomputed','published records only; current source is not the frozen training version'], 'source_sha256':sources})
    files = sorted(p for p in OUT.rglob('*') if p.is_file() and p.name != 'artifact_checksums.sha256')
    (OUT / 'artifact_checksums.sha256').write_text(''.join(sha(p) + '  ' + p.relative_to(OUT).as_posix() + '\n' for p in files), encoding='utf-8')
    print('Prepared validation-only report:', OUT)


if __name__ == '__main__':
    main()
