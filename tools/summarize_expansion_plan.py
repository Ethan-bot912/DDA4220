"""Prepare a new event-disjoint proposal and measured resource budget, no training."""
from pathlib import Path
from collections import Counter
import csv,datetime,json,math,shutil,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/expansion_planning_20261015'
REPORT=ROOT/'reports'/'expansion_planning_20261015'
LABELS=['no_damage','minor','major','destroyed']
def load(f):return json.loads((OUT/f).read_text(encoding='utf-8-sig'))
def tab(headers,rows):return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def gib(x):return x/2**30
def proposal():
 inv=load('inventory.json')
 if inv['label_parse_errors'] or inv['missing_components'] or inv['duplicate_member_names']:raise ValueError('Inventory anomalies need review before proposal')
 events=inv['by_event']
 val=['moore-tornado','hurricane-matthew','portugal-wildfire']
 test=['nepal-flooding','hurricane-florence','santa-rosa-wildfire']
 assert all(e in events for e in val+test)
 train=sorted(set(events)-set(val)-set(test))
 assert {'joplin-tornado','lower-puna-volcano'}<=set(train)
 assign={e:s for s,es in [('train',train),('val',val),('test',test)] for e in es}
 tiles=load('tile_inventory.json');seen={};conflicts=[]
 for t in tiles:
  key=(t['event'],t['tile_id']);s=assign[t['event']]
  if key in seen and seen[key]!=s:conflicts.append(key)
  seen[key]=s
 assert not conflicts
 counts={}
 for s,es in [('train',train),('val',val),('test',test)]:
  c=Counter()
  for e in es:c.update(events[e]['class_counts'])
  counts[s]=dict(events=es,event_count=len(es),building_count=sum(c.values()),class_counts=dict(c),tiles=sum(events[e]['complete_pairs'] for e in es),bytes=sum(events[e]['bytes'] for e in es))
  assert all(c[k]>0 for k in LABELS)
 p=dict(status='proposal_only_not_applied_to_existing_experiment',name='expanded_event_disjoint_v1',original_test_preserved='nepal-flooding',original_val_preserved='moore-tornado',train_events=train,val_events=val,test_events=test,event_assignment=assign,counts=counts,scope='All archive partitions under a new custom event-disjoint project protocol; not the official benchmark split',cross_split_event_overlap=0,cross_split_named_tile_overlap=len(conflicts),test_images_used_in_profile=False,test_evaluated=False,full_image_decode_audit_completed=False)
 (OUT/'split_proposal.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
 with (OUT/'event_split_proposal.csv').open('w',newline='',encoding='utf-8-sig') as h:
  w=csv.writer(h);w.writerow(['event','proposed_split','buildings','tiles','archive_partitions','disaster_types'])
  for e in sorted(events):w.writerow([e,assign[e],events[e]['valid_buildings'],events[e]['complete_pairs'],json.dumps(events[e]['partitions']),json.dumps(events[e]['metadata_types'])])
 print(json.dumps(p,indent=2));return p

def report():
 inv=load('inventory.json');p=load('split_proposal.json');prof=load('profile.json');hw=load('hardware.json');memsample=load('manifest_memory_sample.json')
 REPORT.mkdir(parents=True,exist_ok=True)
 old_extract=json.loads((ROOT/'dataset/pilot_tier3_4events/extraction_summary.json').read_text())
 free=shutil.disk_usage(ROOT).free
 expanded=inv['uncompressed_regular_bytes'];reuse=old_extract['bytes_written'];increment=max(0,expanded-reuse)
 oldmanifest=(ROOT/'dataset/pilot_tier3_4events/manifest_classification.csv').stat().st_size
 totaln=sum(d['valid_buildings'] for d in inv['by_event'].values())
 manifest_est=oldmanifest*totaln/81506
 # 3 GiB for metadata/exports/checkpoints and 20 GiB operational free-space reserve.
 reserve=20*2**30;misc=3*2**30+manifest_est
 required=increment+misc+reserve
 budget=dict(archive_bytes=inv['compressed_bytes'],full_extracted_bytes=expanded,existing_four_event_bytes=reuse,incremental_extraction_bytes_if_reused=increment,free_bytes_now=free,metadata_manifest_estimated_bytes=manifest_est,misc_output_budget_bytes=3*2**30,safety_reserve_bytes=reserve,additional_free_space_required_bytes=max(0,required-free),desired_free_space_before_expansion_bytes=required)
 timing=[]
 for r in prof['results']:
  n=p['counts']['train']['building_count'];v=p['counts']['val']['building_count'];t=p['counts']['test']['building_count']
  train_batches=math.ceil(n/32);val_batches=math.ceil(v/32)
  epoch=train_batches*r['train_batch']['mean_seconds']+val_batches*r['eval_batch']['mean_seconds']
  old=json.loads((ROOT/f'output/resnet18/{r["mode"]}_seed42.json').read_text())
  old_model=20*(math.ceil(17552/32)*r['train_batch']['mean_seconds']+math.ceil(22372/32)*r['eval_batch']['mean_seconds'])
  calibration=max(1,old['training_seconds']/old_model)
  base=20*epoch*calibration
  timing.append(dict(mode=r['mode'],measured_expanded_epoch_seconds=epoch,historical_calibration_multiplier=calibration,baseline_20_epoch_seconds=base,planning_20_epoch_lower_seconds=base*1.2,planning_20_epoch_upper_seconds=base*1.8,independent_val_seconds=math.ceil(v/32)*r['eval_batch']['mean_seconds']*calibration,final_test_seconds_if_later_authorized=math.ceil(t/32)*r['eval_batch']['mean_seconds']*calibration))
 budget['timing']=timing
 budget['two_models_one_seed_lower_hours']=sum(r['planning_20_epoch_lower_seconds'] for r in timing)/3600
 budget['two_models_one_seed_upper_hours']=sum(r['planning_20_epoch_upper_seconds'] for r in timing)/3600
 budget['three_seed_lower_hours']=budget['two_models_one_seed_lower_hours']*3
 budget['three_seed_upper_hours']=budget['two_models_one_seed_upper_hours']*3
 (OUT/'resource_budget.json').write_text(json.dumps(budget,indent=2),encoding='utf-8')
 rows=[]
 for event,d in sorted(inv['by_event'].items()):
  rows.append([event,p['event_assignment'][event],'/'.join(d['metadata_types']),d['complete_pairs'],d['valid_buildings'],*[d['class_counts'].get(k,0) for k in LABELS],f'{gib(d["bytes"]):.2f}'])
 text='''# 全量数据扩展准备与资源预算

日期：2026年10月15日。状态：前三项准备工作完成；未启动正式扩大实验，未评估封存测试集。

## 结论

本次在本机对六分卷进行了完整流式盘点，读取标签统计有效建筑并完成官方归档校验；提出新的事件互斥划分；使用拟定训练事件的少量真实影像进行一次性训练与前向测速。没有全量提取，没有保存测速模型权重，也没有修改原四事件实验数据或结果。

'''
 text+=f'完整归档含 **{len(inv["by_event"])} 个事件、{sum(d["complete_pairs"] for d in inv["by_event"].values()):,} 对影像 tile、{totaln:,} 条可用四分类建筑记录**。完整解压文件体积 **{gib(expanded):.2f} GiB**，六分卷自身 **{gib(inv["compressed_bytes"]):.2f} GiB**。当前 D 盘剩余 **{gib(free):.2f} GiB**。'+('当前磁盘容量已足够，仍需预留输出与运行余量。' if free>=required else '当前磁盘容量不足，扩大提取前需先释放空间。')+'\n\n'
 text+=f'在复用已提取四事件、保留分卷、预留元数据输出和 20 GiB 安全余量的条件下，建议开始扩大提取前空出 **{gib(required):.1f} GiB**；相对于当前空间，还需增加约 **{gib(max(0,required-free)):.1f} GiB**。这些值不包含额外保存完整建筑裁剪缓存。\n\n'
 text+=f'当前机器两模型各20轮、一个seed的预算为 **{budget["two_models_one_seed_lower_hours"]:.1f} 至 {budget["two_models_one_seed_upper_hours"]:.1f} 小时**，约 **{budget["two_models_one_seed_lower_hours"]/24:.1f} 至 {budget["two_models_one_seed_upper_hours"]/24:.1f} 天**。三seed约 **{budget["three_seed_lower_hours"]:.1f} 至 {budget["three_seed_upper_hours"]:.1f} 小时**。这是短时实测结合历史记录的规划区间，不是全量任务已完成的用时，也不是统计置信区间。\n\n'
 text+='## 1 归档盘点与审计范围\n\n'
 text+=f'整体 SHA1 为 {inv["official_sha1"]}，与既定官方值一致。扫描 {inv["members_seen"]:,} 个 tar 成员，耗时 {inv["elapsed_seconds"]/60:.1f} 分钟。结构缺失 {len(inv["missing_components"])}，重复成员名 {len(inv["duplicate_member_names"])}，标签解析错误 {len(inv["label_parse_errors"])}，不同归档分区同名事件与tile重叠 {len(inv["cross_archive_partition_tile_overlap"])}。\n\n'
 text+='“可用建筑”指有完整四组件的tile中、损毁标签可映射到四类且几何框有限并非退化的记录。统计排除了未知标签和无效框，但本次没有对所有影像逐文件解码、全量图像配准审计或跨tile建筑UID去重。因此它是扩大实验前的数据普查，不能替代正式提取后的manifest与影像审计。\n\n'
 text+=tab(['归档分区','配对tile','有效建筑','未知标签','无效框','体积GiB'],[[k,d['complete_pairs'],d['valid_buildings'],d['unknown'],d['invalid_bbox'],f'{gib(d["bytes"]):.2f}'] for k,d in sorted(inv['by_archive_partition'].items())])+'\n\n'
 text+='## 2 全部事件及拟定用途\n\n'
 text+=tab(['事件','拟定划分','标签灾害类型','tile对','有效建筑','无损毁','轻度','重度','摧毁','GiB'],rows)+'\n\n'
 text+='灾害类型来自标签 metadata，而不是只按事件名称猜测。部分飓风事件被标记为 flooding，另一些标记为 wind；报告保留原标签类型。\n\n'
 text+='## 3 事件互斥划分方案\n\n'
 text+=tab(['集合','事件数','建筑数','占全部比例','无损毁','轻度','重度','摧毁'],[[s,d['event_count'],d['building_count'],f'{100*d["building_count"]/totaln:.2f}%',*[d['class_counts'].get(k,0) for k in LABELS]] for s,d in p['counts'].items()])+'\n\n'
 for s in ['train','val','test']:text+=f'**{s}**：'+ '、'.join(p[s+'_events'])+'。\n\n'
 text+='继续将 joplin-tornado 和 lower-puna-volcano 留在训练，将 Moore 留在验证，将 Nepal 留在封存测试。额外的验证与测试事件增加风灾、洪水与火灾场景覆盖；同一事件的所有归档分区统一分配，按事件与命名tile检查跨集合重叠为0。三个集合均包含四类建筑，但并非每个事件都包含四类。\n\n'
 text+='这是覆盖归档全部分区的**自定义事件划分**，不是官方 train/test benchmark 协议。原归档 hold/test 目录内的样本也按事件重新归属；因此未来成绩不能写成官方测试集成绩。若课程要求遵循官方分区，需要改用另一套协议，不能混用两种结果。该提案已保存独立文件，未替换旧manifest。\n\n'
 text+='划分依据是事件身份、标签分布与数据规模，没有使用测试模型分数。测试标签仅用于普查；本次测速只使用拟定train事件，不使用拟定val或test事件。不保证每种灾害类型都在三个集合中出现：单事件类型无法同时实现按事件隔离与全类型覆盖，最终需按事件和类别解释泛化范围。\n\n'
 text+='## 4 短时实测方法\n\n'
 text+=f'只提取每事件至多2对tile，本次保存样本总计 {gib(inv["sample_bytes"]):.3f} GiB，硬上限2 GiB。完整普查结束后才测速，避免归档扫描争用磁盘。由拟定训练事件构建容量清单，共 {prof["train_rows_available"]:,} 条抽样建筑。两组使用相同样本与顺序，ResNet18随机初始化、batch32、crop128、context16、AdamW、float32、workers0与确定性设置。\n\n'
 text+='每模式执行36个训练batch和36个推理batch，前4个各作预热，后32个用于统计，每个阶段测量1024栋建筑。训练模型仅存在于内存，用于测量前向、反向与优化器开销，随后销毁，不保存checkpoint、不报告准确率或F1，也不用于后续正式初始化。推理阶段用训练事件样本模拟验证的读取与前向成本，不是对验证或测试集合做性能评估。\n\n'
 text+=tab(['模式','训练batch秒','其中IO秒','其中计算秒','推理batch秒','训练峰值allocated MiB','训练峰值reserved MiB'],[[r['mode'],f'{r["train_batch"]["mean_seconds"]:.4f}',f'{r["train_io"]["mean_seconds"]:.4f}',f'{r["train_compute"]["mean_seconds"]:.4f}',f'{r["eval_batch"]["mean_seconds"]:.4f}',f'{r["train_peak_allocated_bytes"]/2**20:.1f}',f'{r["train_peak_reserved_bytes"]/2**20:.1f}'] for r in prof['results']])+'\n\n'
 text+='IO时间包含真实GeoTIFF裁剪及DataLoader取batch，计算时间包含输入拼接、设备传输和GPU同步。B1当前实现也读取pre与post后仅使用post。样本只覆盖少量tile，缓存命中通常好于完整训练，因此不能把短时吞吐直接当成全量长期吞吐。\n\n'
 text+=f'本次训练batch中读取与裁剪占比约 {min(r["train_io"]["mean_seconds"]/r["train_batch"]["mean_seconds"] for r in prof["results"])*100:.1f}% 至 {max(r["train_io"]["mean_seconds"]/r["train_batch"]["mean_seconds"] for r in prof["results"])*100:.1f}%。这表明当前流水线有较强的数据处理瓶颈；先优化读取与缓存，比直接推断换GPU能按比例提速更有依据。\n\n'
 text+='## 5 时间预算与假设\n\n'
 text+='计算式：每轮时间 = ceil(训练建筑数/32)×训练batch均值 + ceil(验证建筑数/32)×推理batch均值；正式预算按20轮、每轮完整验证计算。先用相同公式回代旧17,552训练和22,372验证的20轮运行，取历史实际耗时与回代耗时之比，且校正系数不小于1；再在校正后的新规模时间上乘1.2至1.8，给扩大文件数量、冷缓存、随机IO和持续负载留余量。\n\n'
 text+=tab(['模式','短测推算每轮分钟','历史校正系数','20轮规划小时下限','上限'],[[r['mode'],f'{r["measured_expanded_epoch_seconds"]/60:.1f}',f'{r["historical_calibration_multiplier"]:.2f}',f'{r["planning_20_epoch_lower_seconds"]/3600:.1f}',f'{r["planning_20_epoch_upper_seconds"]/3600:.1f}'] for r in timing])+'\n\n'
 text+=f'若完全按短测小样本速度直接外推，两模型20轮合计约 {sum(r["measured_expanded_epoch_seconds"]*20 for r in timing)/3600:.1f} 小时，这是明显乐观的暖缓存情景。旧四事件长期运行比这种回代快照慢约2.4至2.5倍，因此采用历史校正并增加余量后的92至138小时作为当前实现的安排依据。将来完成有界缓存或IO优化后，需要重新测量，不应同时沿用旧校正与宣称新的提速。\n\n'
 text+=f'单seed两模型合计 {budget["two_models_one_seed_lower_hours"]:.1f}–{budget["two_models_one_seed_upper_hours"]:.1f} 小时；三seed为其三倍，即 {budget["three_seed_lower_hours"]:.1f}–{budget["three_seed_upper_hours"]:.1f} 小时。此为本机独占连续运行的占用时间近似，不是纯GPU计算小时。包括逐轮验证，不包括完整提取与审计、故障重跑或最后独立评估。\n\n'
 text+=f'恢复权重独立验证每模型按短测和历史校正约 {min(r["independent_val_seconds"] for r in timing)/60:.1f}–{max(r["independent_val_seconds"] for r in timing)/60:.1f} 分钟；未来如获授权，最终测试每模型约 {min(r["final_test_seconds_if_later_authorized"] for r in timing)/60:.1f}–{max(r["final_test_seconds_if_later_authorized"] for r in timing)/60:.1f} 分钟，均未加长期IO余量。实际提取写盘速度未测量，本次扫描速度不能直接等同于全量解压写盘速度；准备阶段建议另留2至4小时作为提取、构建清单和审计的初步调度预算，具体需在目标磁盘小规模写入后更新。\n\n'
 text+='## 6 算力与内存\n\n'
 text+=f'GPU为 {prof["gpu"]}，总显存 {prof["gpu_total_bytes"]/2**30:.2f} GiB；CPU为 {hw["cpu"]["Name"]}，{hw["cpu"]["NumberOfCores"]}核{hw["cpu"]["NumberOfLogicalProcessors"]}线程；物理内存 {hw["memory"]["TotalPhysicalMemory"]/2**30:.2f} GiB。batch32、crop128的小规模训练已实测能运行。扩大样本量通常增加迭代次数，而不是按样本总数同比增加单batch显存；但系统RAM和影像句柄会随实现而变化。\n\n'
 text+=f'测速开始时系统可用RAM约 {gib(prof["ram_before"]["available"]):.2f} GiB，测速中最低约 {gib(min(r["system_available_ram_min_bytes"] for r in prof["results"])):.2f} GiB，说明物理16GiB并不等于任务可独占16GiB。正式运行前应检查其他应用占用，而不是只看机器标称内存。\n\n'
 text+='本机显存可继续支持相同结构与batch的实验；不需要仅因增加数据量就购买更大显存GPU。实际速度可能受GeoTIFF裁剪和磁盘读取限制，换更快GPU的加速比不能按GPU算力规格直接推断。若使用其他设备，必须在目标设备上复测IO与每batch时间；本报告没有实测云GPU，未给出虚构的云训练时长或租赁价格。\n\n'
 text+=f'从旧清单用固定种子蓄水抽样2,048条完整几何记录，单条Python字典及其值平均约 {memsample["average_python_row_bytes"]:.0f} 字节。按新建筑数外推，单次完整CSV载入约 {gib(totaln*memsample["average_python_row_bytes"]):.2f} GiB；构建验证Dataset时还保留训练Dataset，保守瞬时对象合计约 {gib((totaln+p["counts"]["train"]["building_count"])*memsample["average_python_row_bytes"]):.2f} GiB。这个外推不含影像句柄、GDAL缓存、模型和系统，也不能保证不同事件几何字符串长度一致。\n\n'
 text+='当前XBDPairedDataset对每幅打开的影像保留句柄，缺少有界缓存淘汰，全量文件数明显大于旧实验。正式扩展前应加有界LRU/关闭策略，并逐步测量句柄与RAM；保持workers0作为起点，不应直接开24个worker。全量CSV含几何字符串，Python对象开销也需要在提取后测量；本次未加载全量正式Dataset，不能保证16GiB内存在未修正缓存时足够。\n\n'
 text+='## 7 磁盘预算\n\n'
 text+=tab(['项目','GiB','性质'],[['六个原始分卷',f'{gib(inv["compressed_bytes"]):.2f}','已占用 保留'],['完整归档解压文件',f'{gib(expanded):.2f}','根据tar成员size精确求和'],['已提取四事件',f'{gib(reuse):.2f}','已占用 可复用'],['新增解压净需求',f'{gib(increment):.2f}','复用四事件前提'],['完整manifest估计',f'{gib(manifest_est):.2f}','按原CSV每建筑字节线性估计'],['其他元数据结果及模型预算','3.00','规划余量'],['磁盘安全余量','20.00','保留空闲'],['扩大开始前建议可用空间',f'{gib(required):.2f}','新增需求加输出和安全余量'],['当前可用',f'{gib(free):.2f}','报告生成时快照'],['还需增加可用空间',f'{gib(max(0,required-free)):.2f}','目标减当前']])+'\n\n'
 text+=f'如果不复用旧四事件、而在新目录重复提取全部数据，需要再多 {gib(reuse):.2f} GiB。额外创建合并归档还会再占 {gib(inv["compressed_bytes"]):.2f} GiB，应继续用六分卷流式读取。当前不删除任何已有分卷、数据、环境或结果；残留下载临时文件也未自动处理。\n\n'
 text+=f'如预存每栋建筑的灾前灾后128×128无压缩RGB裁剪，全部建筑约需 {gib(totaln*2*3*128*128):.1f} GiB（uint8）或 {gib(totaln*2*3*128*128*4):.1f} GiB（float32），还未计文件开销。这些是额外缓存而非原始GeoTIFF的替代保证，当前不建议在磁盘紧张时生成。\n\n'
 text+='## 8 正式扩大前的具体条件\n\n'
 text+='1. 确定采用自定义事件划分，并保存版本及校验信息。\n2. 按最新磁盘余量安排数据路径，保留至少20GiB空闲，继续保留原实验；旧的32.8GiB空间快照已由用户清理后约274.4GiB的新快照取代。\n3. 将清单构建从当前固定tier3扩展为多归档分区，并处理可能的同名记录。\n4. 限制影像句柄缓存，测量全量清单的RAM占用；不改变网络与核心训练协议。\n5. 完成扩大数据的实际提取、manifest生成、完整结构和影像审计及样本人工检查。\n6. 在新输出目录运行更长的容量试跑，确认时间估计与资源稳定，再由用户决定正式扩大实验。\n\n'
 text+='本次仅完成盘点、划分提案和短时资源测量。没有进行20轮全量训练、没有追加三seed、没有测试集性能评估，也没有覆盖旧权重。\n\n'
 text+='## 9 证据文件\n\n'
 for name in ['inventory.json','tile_inventory.json','event_inventory.csv','split_proposal.json','event_split_proposal.csv','hardware.json','profile.json','capacity_manifest.csv','resource_budget.json','manifest_memory_sample.json']:
  text+=f'- {OUT/name}\n'
 (REPORT/'全量扩展准备与资源预算.md').write_text(text,encoding='utf-8')
 for name in ['event_inventory.csv','event_split_proposal.csv']:shutil.copy2(OUT/name,REPORT/name)
 print(json.dumps(budget,indent=2))
 print('Report:',REPORT/'全量扩展准备与资源预算.md')

if __name__=='__main__':
 if '--proposal' in sys.argv:proposal()
 else:report()
