from pathlib import Path
from html import escape
import re, shutil, json

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'repo'
SKILL = Path('C:/Users/SXF-Admin/.codex/skills/kelip-slide')
MEDIA = OUT / 'media'
MEDIA.mkdir(parents=True, exist_ok=True)
assets = {
    'training.png': 'paper-latex/figures/hjepa_training.png',
    'planning.png': 'paper-latex/figures/hierarchical_planning.png',
}
for name, src in assets.items(): shutil.copy2(ROOT / src, MEDIA / name)
if not (OUT / 'render_preview.py').exists():
    shutil.copy2(SKILL / 'assets/render_preview.py', OUT / 'render_preview.py')
slides, notes = [], []

def add(sec, title, body='', source='', note=''):
    n = len(slides) + 1
    title_text = re.sub('<[^>]+>', '', title)
    content = f'<h1>{title}</h1>' + body
    slides.append(f'<section class="slide" data-sec="{escape(sec)}" data-title="{escape(title_text)}">{content}</section>')
    notes.append((n, sec, title_text, source, note))

def fig(name,alt): return f'<div class="fig card"><img src="media/{name}" alt="{escape(alt)}"></div>'
def txt(x,y,s,size=34,color='#1D1D1F',anchor='middle',weight=500):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{max(size, 30)}" fill="{color}" font-weight="{weight}">{escape(str(s))}</text>'
def rect(x,y,w,h,accent=False):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{"#E8F0FE" if accent else "#F5F5F7"}"'+(' stroke="#0071E3" stroke-width="2"' if accent else '')+'/>'
def box(x,y,w,h,label,desc='',accent=False):
    return rect(x,y,w,h,accent)+txt(x+w/2,y+h/2+(-8 if desc else 12),label,36,'#0071E3' if accent else '#1D1D1F',weight=600)+(txt(x+w/2,y+h/2+39,desc,30,'#6E6E73') if desc else '')
def arrow(x1,y1,x2,y2,blue=False,dash=False):
    return f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{"#0071E3" if blue else "#8E8E93"}" stroke-width="3" marker-end="url(#{"arB" if blue else "ar"})"'+(' stroke-dasharray="8 6"' if dash else '')+'/>'
def svg(content): return '<div class="diagbox pad"><svg class="sv" viewBox="0 0 1600 560" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">'+content+'</svg></div>'
def table(headers, rows):
    return '<div class="tblwrap"><table class="tbl left wrap"><thead><tr>'+''.join(f'<th>{x}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{c}</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table></div>'

add('层级编码','H-JEPA 的层级世界模型',fig('training.png','图2：状态逐层编码、动作按跨度汇总，各层预测与正则'),source='论文图 2',note='先看左半。蓝色是状态，橙色是动作。底层图像经同一个 E¹ 编码成状态，不同时间位置画出的多个 E¹ 不是多个独立网络。第2层只取 z⁽¹⁾₀、z⁽¹⁾₂ 等状态，状态 window=1。动作则汇总中间两步的动作嵌入，保证覆盖从当前上层状态到下一个上层状态的完整转移。右侧每层的 F 预测下一状态，以未来观测的编码作为目标，同时对状态集合加 SIGReg。来源 main_release.tex:786–864。')


content=box(30,80,320,130,'64 × 64 RGB','')+box(30,330,320,130,'27 维本体状态','')
content+=box(480,80,360,130,'ViT-Tiny + 投影','全局 CLS → 256 维')+box(480,330,360,130,'本体 MLP','27 → 128 维')
content+=box(1040,195,490,150,'状态 z¹','256 + 128 = 384',True)
content+=arrow(350,145,470,145)+arrow(350,395,470,395)+arrow(840,145,1030,245)+arrow(840,395,1030,295)
add('层级编码','Ant 的状态向量是<span class="acc">256 + 128</span>',svg(content),source='ant.yaml · L1 配置',note='对一帧 Ant 观测，64×64 RGB 使用4×4 patch，产生16×16个视觉 patch，ViT-Tiny的全局 CLS 再投影为256维。27维本体感知通过MLP成为128维，两者拼成384维状态。27维由 qpos 去掉开头的全局xy后再拼qvel；全局位置需要从图像里恢复。高层接收到的是384维向量，并非完整patch序列。源码：config/train/base/ant.yaml 中 img_size、patch_size、pixel_encoder、proprio_encoder 与 wm。')


content = ''
for row, (level, stride, span) in enumerate([(1, 1, 5), (2, 2, 10), (3, 4, 20)]):
    y = 80 + row * 160
    content += txt(95, y + 10, f'L{level}', 40, weight=600)
    content += txt(95, y + 57, f'{span} 步', 30)
    for raw_index in range(0, 5, stride):
        x = 350 + raw_index * 255
        local_index = raw_index // stride
        label = 'z' + '¹²³'[level - 1] + '₀₁₂₃₄'[local_index]
        content += box(x - 85, y - 40, 170, 80, label, accent=level > 1)
        if level < 3 and raw_index % (stride * 2) == 0:
            content += arrow(x, y + 50, x, y + 110)
            content += txt(x + 77, y + 92, 'E' + '²³'[level - 1], 30)
content += txt(800, 545, 'w = 1       s = 2       每个状态均为 384 维', 36, '#0071E3', weight=600)
add('层级编码', '状态逐点编码，时间步长逐层翻倍', svg(content), source='main_release.tex 式1，ant_hjepa_l3.yaml', note='w=1表示上层编码器一次读取一个下层状态，s=2表示相邻上层状态间隔两个下层步。图中五列对应环境步0、5、10、15、20。z²₀=E²(z¹₀)，z²₁=E²(z¹₂)，z³₀=E³(z²₀)。各层状态均为384维，预测跨度为5、10、20环境步。高层状态MLP学习适合本层预测跨度的表征。')


content = box(20, 45, 370, 140, 'L1 动作块', '5 × 8 = 40 维')
content += box(620, 45, 370, 140, '动作嵌入', '40 → 384 维')
content += box(1200, 45, 370, 140, '预测器 F¹')
content += arrow(390, 115, 610, 115) + arrow(990, 115, 1190, 115)
content += box(20, 340, 370, 150, '两个下层动作嵌入', '2 × 384')
content += box(620, 340, 370, 150, '宏动作编码 A²', '4 维宏动作', True)
content += box(1200, 340, 370, 150, '动作嵌入 → F²', '4 → 384 维')
content += arrow(390, 415, 610, 415) + arrow(990, 415, 1190, 415)
add('层级编码', '动作汇总覆盖完整的时间间隔', svg(content), source='models/jepa.py encode，models/encoders/seq_encoder.py，ant_hjepa_l3.yaml', note='L1把5个8维真实控制拼接成40维，再经action_embed得到384维动作条件。L2的action_encoder读取两个下层动作嵌入，以单层Transformer和CLS汇总为4维宏动作，随后action_embed将其映射为384维预测器条件。宏动作各维由学习决定。L3采用相同的递归构造。规划时直接优化4维宏动作。')


content = box(30, 40, 390, 130, '当前状态 + 动作', 'zₜ + aₜ')
content += box(600, 40, 330, 130, '预测器 F')
content += box(1110, 40, 440, 130, '预测状态', 'ẑₜ₊₁', True)
content += arrow(420, 105, 590, 105) + arrow(930, 105, 1100, 105)
content += box(30, 275, 390, 130, '真实未来观察', 'oₜ₊₁')
content += box(600, 275, 330, 130, '本层编码路径 E', '从观察逐层编码')
content += box(1110, 275, 440, 130, '目标状态', 'zₜ₊₁', True)
content += arrow(420, 340, 590, 340) + arrow(930, 340, 1100, 340)
content += txt(800, 520, 'Lpred = mean[(ẑₜ₊₁ − zₜ₊₁)²]', 44, '#0071E3', weight=600)
add('训练目标', '预测未来状态，对齐真实未来的编码', svg(content), source='图2(b)，main_release.tex 式2，h_jepa/hjepa_utils.py', note='Fig.2(b)的F接收本层状态和动作条件，输出下一状态预测ẑ。目标z由真实未来观察沿编码路径产生。上标表示层，下标表示本层时间，有帽子的是预测量。预测与目标在同一层空间比较。示意预测[0.2,0.7]、目标[0.4,0.6]时，MSE=(0.04+0.01)/2=0.025。实际Ant向量为384维。仿真每层使用三个teacher-forced预测位置。图下方的SIGReg约束状态集合的分布，下一页展开。')


content=txt(370,65,'只有预测误差',33,weight=600)+txt(1230,65,'再加 SIGReg',33,'#0071E3',weight=600)
content+=rect(130,100,480,330)+rect(990,100,480,330,True)
for i in range(30):
    content+=f'<circle cx="{370+(i%5-2)*2}" cy="{265+(i//5-2)*2}" r="5" fill="#6E6E73"/>'
    x=1230+((i*73)%290-145);y=265+((i*113)%230-115)
    content+=f'<circle cx="{x}" cy="{y}" r="6" fill="#0071E3" opacity=".65"/>'
content+=txt(370,475,'所有状态一样 → 预测也容易',27,'#6E6E73')+txt(1230,475,'随机投影后，逼近标准正态',27,'#6E6E73')+txt(800,540,'L = Σℓ [ Lpred⁽ℓ⁾ + λℓ · SIGReg(Z⁽ℓ⁾) ]',34,weight=600)
add('训练目标','SIGReg 防止状态表征坍塌',svg(content),source='式 (3) · SIGReg 示意',note='如果所有图像都被编码成同一个常数，预测器输出同一个常数就能把MSE降得很低，但表征对控制毫无用处。SIGReg在多个随机方向上检查嵌入分布，让投影分布接近各向同性高斯，避免总体坍塌。仿真使用1024个随机投影，Ant各层λ=0.09。图上的散点是解释原理的示意，不是把真实高维向量做PCA后的可视化。关键限制：边缘分布有方差不等于保留了轨迹内运动；场景之间的差异也能提供方差，这正是DROID需要IDM的原因。')


content=box(30,210,240,110,'观测序列','')+box(380,210,250,110,'E¹','')+box(750,210,250,110,'E²','')+box(1120,210,250,110,'E³','')
content+=arrow(270,265,370,265)+arrow(630,265,740,265)+arrow(1000,265,1110,265)
for x,lab in [(505,'L¹'),(875,'L²'),(1245,'L³')]: content+=box(x-85,45,170,80,lab,accent=True)+arrow(x,200,x,135)
content+=arrow(1245,365,510,365,True)+arrow(875,425,510,425,True)+txt(800,520,'Ltotal = L¹ + L² + L³',40,'#0071E3',weight=600)
add('训练目标','各层联合训练，高层梯度传回低层',svg(content),source='主结果：联合端到端训练',note='所有层的损失一起反向传播，所以上层要求“更可预测”的表征，也会塑造低层编码器。主结果没有EMA教师、没有对latent预测目标stop-gradient、没有先冻结第1层再训练第2层。图只突出状态编码链；每层的动作编码器与预测器也被对应损失更新。DROID同样主要使用端到端训练。论文里确实还出现stagewise变体，但只用于数据专门化、系数搜索和部分解码计划分析，不能用这些局部图替代主训练流程。来源附录C.1/C.2、表5。')


content=box(30,60,320,100,'zₜ','当前状态')+box(30,260,320,100,'zₜ₊₁','下一状态')+box(530,160,360,130,'逆动力学 I','从状态变化猜动作',True)+box(1100,160,400,130,'âₜ vs sg[aₜ]')
content+=arrow(350,110,520,200)+arrow(350,310,520,250)+arrow(890,225,1090,225)+txt(800,515,'LIDM = mean[(âₜ − sg[aₜ])²]',40,'#0071E3')
add('训练目标','DROID 用逆动力学保留运动信息',svg(content),source='式 (8) · 附录 K',note='DROID场景每个episode都变，模型可能只记住场景身份：这一信息易预测且有总体方差，但同一视频内手臂运动被忽略。IDM要求从相邻两个状态还原连接它们的动作，因此状态必须携带动作相关的变化。每层增加γℓ L_idm；上层标签是学到的宏动作，不是原始7维控制，要detach，避免通过改变标签来降低误差。主设置没有对latent预测目标detach，不能混淆。例：二维动作标签[0.2,-0.1]，预测[0.1,-0.1]，逐维MSE=0.005；这只是算术示意。DROID L1/L2的IDM权重分别100/50，SIGReg分别0.08/0.16，上层另有动作SIGReg 0.005。')


add('层级规划','高层的<span class="acc">预测状态</span>，成为低层子目标',fig('planning.png','图3：上层rollout提供latent子目标，下层预测投影后匹配'),source='论文图 3 · 式 (4–6)',note='图左下当前图像和右下目标图像都向上编码。上层先优化动作，让最后一个预测靠近最终目标；随后把第一预测状态向下交给低层。低层不是直接拿高层动作解码，而是通过自己的预测器展开候选动作，把末端状态经E²映射到高层空间，再和子目标比较。这样比较的是同一层的同一时刻含义。图中两层、K2=1、stride2、window1对应低层预测两步去追上层一步。来源main_release.tex:1033–1112。')


content=box(30,190,290,130,'候选动作 A','本轮待优化变量',True)+box(440,190,310,130,'世界模型 rollout','模型参数固定')+box(870,190,310,130,'目标距离 J(A)','终点或子目标')+box(1300,190,270,130,'AdamW 更新','更新候选动作',True)
content+=arrow(320,255,430,255)+arrow(750,255,860,255)+arrow(1180,255,1290,255)
content+='<path d="M 1435 335 L 1435 435 L 175 435 L 175 335" fill="none" stroke="#0071E3" stroke-width="3" marker-end="url(#arB)"/>'
content+=txt(800,515,'模型参数固定，梯度优化候选动作',34,'#1D1D1F')
add('层级规划','在世界模型中迭代优化动作',svg(content),source='附录 E · GradientSolver',note='这不是直接调用一个学好的策略网络前向吐动作，而是在世界模型里做测试时优化。多条候选动作从先验分布初始化，通过可微预测器算代价；AdamW最多迭代90次，停滞时可提前结束，最后选择代价最低的候选。模型参数冻结不会阻断输出对输入动作的导数。高层4维宏动作没有天然物理范围，代码保存最近4000个训练宏动作，拟合逐维高斯初始化并按经验分位数裁剪。Ant没有额外动作匹配损失。对应stable_worldmodel/solver中的GradientSolver以及hierarchical_solver.py。')


content=''
for j,(lev,hor,span,lab) in enumerate([('L3',5,20,''),('L2',2,10,'L2 追踪 20 步子目标'),('L1',2,5,'L1 追踪 10 步子目标')]):
    y=95+j*160; content+=txt(100,y,lev,34,weight=600)+txt(100,y+43,f'H={hor}',24,'#6E6E73')
    for k in range(hor):content+=box(270+k*220,y-52,180,85,str((k+1)*span)+' 步',accent=k==0)
    content+=txt(1490,y+80,lab,25,'#6E6E73',anchor='end')
content+='<path d="M 360 138 L 360 166 L 580 166 L 580 193" fill="none" stroke="#0071E3" stroke-width="3" marker-end="url(#arB)"/>'
content+='<path d="M 360 298 L 360 326 L 580 326 L 580 353" fill="none" stroke="#0071E3" stroke-width="3" marker-end="url(#arB)"/>'
add('层级规划','Ant 初轮按 <span class="acc">5 / 2 / 2</span> 步逐层规划',svg(content),source='Ant 三层 · HierarchicalSolver 的初始调度',note='真实Ant三层配置不是让每一层都展开到同一个最终时间。初轮L3用5步，每步20环境步，覆盖100环境步。只把第一个20步后的预测交给L2，L2用两个10步预测追它。再只取L2第一预测，L1用两个5步动作块追10步后的子目标。因此本轮实际执行动作覆盖10环境步。YAML中L1 horizon=4是接近评估预算尾部时的衰减调度起点，初期使用pre_decay_horizon=2。仅照读horizon字段会画错。来源config/eval/ant_l3.yaml、hierarchical_solver.py:241–278。')


content = box(20, 90, 390, 140, 'L2 两步预测', 'ẑ²₂')
content += box(610, 90, 360, 140, '上层编码器 E³', '映射到 L3 空间')
content += box(1170, 90, 390, 140, '投影预测', 'E³(ẑ²₂)', True)
content += arrow(410, 160, 600, 160) + arrow(970, 160, 1160, 160)
content += box(1170, 335, 390, 130, '上层子目标', 'ẑ³₁,*', True)
content += txt(600, 400, '两者均对应 20 环境步', 36)
content += txt(800, 540, 'J₂ = ‖ E³(ẑ²₂) − ẑ³₁,* ‖²', 44, '#0071E3', weight=600)
add('层级规划', '在同一层空间比较预测与子目标', svg(content), source='main_release.tex 式5–6，hierarchical_solver.py 投影代价', note='L2两次10步预测与L3第一次20步预测在时间上对齐。L2终点经E³映射后，两侧还处于相同的表征空间。示意子目标[0.3,0.4]，投影预测[0.2,0.5]的平方距离为0.02，另一候选[0.29,0.42]为0.0005。数字是两维算术示意。选好L2计划后，将其首个10步预测交给L1。')


body = table(['层级', '候选数 × 时域 × 动作维度', '每步覆盖'], [
    ['L3', '400 × 5 × 4', '20 环境步'],
    ['L2', '100 × 2 × 4', '10 环境步'],
    ['L1', '100 × 2 × 40', '5 环境步'],
])
body += '<div class="execution">选中 L1 动作块 <b>2 × 40</b> → 执行 <b>10 × 8</b></div>'
add('层级规划', '把宏动作规划落实为原始控制', body, source='config/eval/ant_l3.yaml，stable_worldmodel/policy.py get_action', note='这是Ant三层初轮配置，省略batch=1。顶层搜索400条候选，下两层各100条。4维为可规划宏动作，40维为5个8维控制的拼接。求解器选出一条L1序列，策略保留两个动作块，重排为10个8维控制，完成反归一化后执行。')


content = box(20, 170, 320, 150, '当前观察', 'RGB + 本体状态')
content += box(440, 170, 320, 150, '三级规划', 'L3 → L2 → L1', True)
content += box(860, 170, 320, 150, '执行动作', '10 个环境步')
content += box(1280, 170, 300, 150, '新的观察', '下一轮起点')
content += arrow(340, 245, 430, 245) + arrow(760, 245, 850, 245) + arrow(1180, 245, 1270, 245)
content += '<path d="M 1430 335 L 1430 460 L 180 460 L 180 335" fill="none" stroke="#0071E3" stroke-width="3" marker-end="url(#arB)"/>'
add('闭环执行', '执行后重新观察，再次规划', svg(content), source='stable_worldmodel/policy.py get_action，main_release.tex 附录E', note='执行10个环境步后读取新的图像和本体状态，再编码到各层，从真实新状态朝同一目标重新规划。实际执行与预测之间的偏差通过重观察修正。当前实现每层规划history默认1，下一轮候选依据solver先验重新初始化。')


assert len(slides) == 14

template = (SKILL / 'assets/deck-template.html').read_text(encoding='utf-8')
head = template[:template.index('<div id="stage">')]
head = re.sub(r'<title>.*?</title>', '<title>H-JEPA · Method</title>', head)
custom = """
:root{--font:"PingFang SC","Microsoft YaHei","SF Pro Display","Hiragino Sans GB","Helvetica Neue",Arial,sans-serif;--mono:"Cascadia Code","Consolas",monospace}
.slide{padding:80px 110px 84px}
h1{font-size:70px;line-height:1.15;margin-top:0}
.sv text{font-family:var(--font)}
.tblwrap{margin-top:42px;display:flex;align-items:center}
.tbl{width:100%}
.tbl.left.wrap th{font-size:32px;padding:26px 25px;color:var(--ink)}
.tbl.left.wrap td{font-size:38px;padding:32px 25px;line-height:1.4}
.tbl.left.wrap td:first-child{width:18%}
.execution{font-size:40px;text-align:center;margin:34px 0 24px}
.execution b{color:var(--accent);font-weight:600}
#nav-btn{opacity:0}
#nav-hot:hover ~ #nav-btn,#nav-btn:hover,#nav-btn.open{opacity:1}
#nav-panel{font-size:18px;width:520px}
#nav-panel .sec-title{font-family:var(--font);font-size:17px;letter-spacing:0}
#nav-panel .it .k{font-size:17px}
"""
head = head.replace('</style>', custom + '\n</style>')
defs = '<div id="stage"><svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs><marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="#8E8E93"/></marker><marker id="arB" markerWidth="9" markerHeight="9" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="#0071E3"/></marker></defs></svg>'
tail = template[template.index('<div id="nav-hot">'):]
tail = tail.replace("'← → 翻页 · F 全屏 · C 页码 · 左上角选页'", "''")
tail = tail.replace("  hud.classList.add('show'); setTimeout(()=>{ if(!showCounter) hud.classList.remove('show'); },4000);", '')
(OUT / 'deck.html').write_text(head + defs + '\n'.join(slides) + '</div>\n' + tail, encoding='utf-8')

md = '# H-JEPA Method 逐页讲稿\n\n与精简版14页对应。页面展示机制，来源与数字示例在此补充。\n\n'
for n, section, title, source, note in notes:
    md += f'## {n:02d} · {title}\n\n{note}\n\n依据 {source}。\n\n'
(OUT / '讲稿.md').write_text(md, encoding='utf-8')
manifest = {'slides': len(slides), 'focus': 'method', 'assets': assets, 'external_runtime_dependencies': [], 'source_snapshot': 'user-provided local LaTeX and repository'}
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

description = '# H-JEPA Method 讲解稿\n\n从44页精简为14页，围绕层级编码、训练目标、自顶向下规划和闭环执行展开。\n\n原版及全部素材备份在 `../backups/repo-original-44pages/`，可直接打开其中的 `deck.html`。\n\n## 页码\n\n| 页 | 内容 |\n|---|---|\n'
description += ''.join(f'| {n} | {title} |\n' for n, section, title, source, note in notes)
description += """
## 使用

Chrome打开 `deck.html`。方向键或空格翻页，G打开目录，F全屏，C切换页码。导航在鼠标移到左上角时出现。所有运行素材在 `media/`，支持离线打开。

## 素材

- `media/training.png` 来自 `../paper-latex/figures/hjepa_training.png`。
- `media/planning.png` 来自 `../paper-latex/figures/hierarchical_planning.png`。
- 其余页面为按论文和代码绘制的SVG，详细出处在 `讲稿.md`。

## 构建与检查

`python -X utf8 build_deck.py` 重建HTML和逐页讲稿。

`render_preview.py deck.html 14 preview 1` 逐页渲染，需要设置CHROME环境变量并使用带Pillow的Python。

浏览器检查记录见 `validation.json`，渲染结果见 `preview/` 和 `预览-全部页面.png`。
"""
(OUT / '说明.md').write_text(description, encoding='utf-8')
print(f'Built {len(slides)} method slides')
