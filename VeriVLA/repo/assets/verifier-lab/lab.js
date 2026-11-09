(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const base = 'https://github.com/shirley430316/VeriVLA/blob/f95ab142d6b5b247b47b36137352eb6289dd86a7/verifier/';
  const source = (file,range) => base + file + '#L' + range;
  const fmt = (v,n=3) => (Math.abs(v)<1e-9 ? 0 : v).toFixed(n);
  const vector = x => '[' + x.map(v=>fmt(v,2)).join(', ') + ']';
  const announce = text => { $('announce').textContent=text; };
  const modules = {
    encoder:{no:'PART 01 / ENCODER',title:'图像 → 视觉 tokens',description:'先用 16 × 16 的小块编码图像，再由视觉 Transformer 交换空间信息。256 × 256 图像得到 256 个位置的特征，每个 768 维。论文使用冻结的预训练编码器；公开类本身只定义结构。',shape:'[B, 3, 256, 256]\n       ↓ PatchEmbed + ViT\n[B, 256, 768]\n       ↓ predictor_embed\n[B, 256, 1024]',file:'encoder.py',range:'138-L174',code:'x = self.patch_embed(x) + self.pos_embed\nfor blk in self.blocks:\n    x = blk(x, ...)\nreturn self.norm(x)'},
    action:{no:'PART 02 / ACTION ENCODING',title:'每个动作成为一个 token',description:'一段候选动作预先分成 5 个子块。示例中，每子块有 10 个动作，每个动作 14 维。线性层逐个映射动作；它保留 10 个 token，不把整个子块压成一个向量。',shape:'[B, 5, 10, 14]\n       ↓ Linear(14, 1024)\n[B, 5, 10, 1024]',file:'ac_predictor.py',range:'63-L70',code:'a = selected_action_encoder(actions)\n# action_dim 支持 7 / 14 / 16\n# 最后一维 → predictor_embed_dim'},
    state:{no:'PART 02 / STATE ENCODING',title:'红色 token 是机器人状态',description:'机器人本体状态经过线性层成为一个 token。若只传入一份当前状态 s₀，代码将它重复到各子块；这是一种输入分支，不是未来状态预测。若传入逐子块状态序列，则直接使用该序列。',shape:'[B, 14] → repeat\n[B, 5, 14] → Linear + unsqueeze\n[B, 5, 1, 1024]',file:'ac_predictor.py',range:'157-L219',code:'s = state_encoder(states).unsqueeze(-2)\n# 一个子块对应一个状态 token\n# 训练中的状态约定仍未完整公开'},
    task:{no:'PART 03 / TASK QUERY',title:'任务 ID → 可学习的向量',description:'从任务向量表中选一行，作为每个子块的进度读取位置。它进入 Transformer 后也生成 Q、K、V；经过多层读取动态信息，最后由进度头输出一个标量。',shape:'task_id\n       ↓ Embedding(40, 1024)\n[B, 1024]\n       ↓ expand across T\n[B, T, 1, 1024]',file:'ac_predictor.py',range:'180-L186',code:'task_term = self.task_query_embedding(tid)\nreturn task_term.view(B, 1, 1, D).expand(B, T, 1, D)'},
    transformer:{no:'PART 04 / SHARED TRANSFORMER',title:'在同一主干里交换信息',description:'每子块拼接 268 个 token。当前源码默认 24 层、16 头、隐藏维度 1024。注意力负责跨 token 读取，FFN 逐 token 变换特征。掩码阻断任务向量进入动态路径。',shape:'X: [B, T×268, 1024]\nQ/K/V: [B, 16, T×268, 64]\nAttention → residual → FFN\nH: [B, T×268, 1024]',file:'attention.py',range:'521-L577',code:'h = h + Attention(LayerNorm(h), mask)\nh = h + FFN(LayerNorm(h))\n# 本段为主干计算的简化写法'},
    latent:{no:'PART 05 / LATENT HEAD',title:'把视觉位置读成未来特征',description:'从共享主干输出中截取 256 个视觉位置。先归一化，再映射回编码器的 768 维特征空间。这个预测 latent 会进入下一子块；它并没有被解码成未来图像。',shape:'[B, T, 256, 1024]\n       ↓ LayerNorm + Linear\n[B, T×256, 768]',file:'ac_predictor.py',range:'268-L273',code:'out = h[:, :, cond_tokens:cond_tokens+HW, :]\nout = self.predictor_proj(self.predictor_norm(out.flatten(1, 2)))'},
    progress:{no:'PART 05 / PROGRESS HEAD',title:'把任务位置读成进度增量',description:'取出每个子块最后的任务 token。名为 value_mlp 的模块实际是一层 Linear，将 1024 维变成 1 维。其直接输入是更新后的任务特征；输出可正可负，不是成功概率。',shape:'[B, T, 1024]\n       ↓ Linear(1024, 1)\n[B, T, 1]\n       ↓ sum over sub-chunks\n每条候选一个分数',file:'ac_predictor.py',range:'268-L280',code:'value_tok = h[:, :, cond_tokens + HW, :]\nvalue_logits = self.value_mlp(value_tok)'}
  };
  function showModule(name) {
    const m=modules[name];
    $('module-no').textContent=m.no;
    $('module-title').textContent=m.title;
    $('module-description').textContent=m.description;
    $('module-shape').textContent=m.shape;
    $('module-code').textContent=m.code;
    $('module-source').textContent='verifier/'+m.file+':'+m.range.replace('-L','–');
    $('module-source').href=source(m.file,m.range);
    document.querySelectorAll('[data-module]').forEach(b=>{
      const active=b.dataset.module===name;
      b.classList.toggle('selected',active); b.setAttribute('aria-pressed',String(active));
    });
  }
  document.querySelectorAll('[data-module]').forEach(b=>b.addEventListener('click',()=>showModule(b.dataset.module)));
  showModule('encoder');

  // Five-step scene illustration. No model or robot is executed here.
  const roll={step:0,candidate:0,task:0,timer:null};
  const candidates=[
    {name:'A · 持续抬起',angles:[20,32,45,58,70,80]},
    {name:'B · 小幅抬起',angles:[20,28,36,38,45,48]},
    {name:'C · 逐渐放下',angles:[20,17,12,8,5,0]}
  ];
  function scene(svg,angle,imagined) {
    const rad=angle*Math.PI/180, ex=88+111*Math.cos(rad), ey=145-111*Math.sin(rad);
    const arm=imagined?'#91a994':'#8b9790', lid=imagined?'#477e6c':'#71837c';
    const tiles=Array.from({length:16},(_,i)=>{
      const tone=Math.round(155+45*Math.sin(i*1.9+angle*.07));
      return '<rect x="'+(249+(i%4)*7)+'" y="'+(30+Math.floor(i/4)*7)+'" width="5" height="5" rx="1" fill="rgb('+tone+','+(tone+14)+','+(tone+7)+')"/>';
    }).join('');
    svg.innerHTML='<defs><pattern id="'+svg.id+'-grid" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M20 0H0V20" fill="none" stroke="#dee5d9" stroke-width=".6"/></pattern></defs>'+
      '<rect x="8" y="11" width="284" height="165" rx="10" fill="url(#'+svg.id+'-grid)"/>'+
      '<path d="M14 163H286" stroke="#c1cdbd"/><ellipse cx="146" cy="164" rx="113" ry="6" fill="#dce5d9"/>'+
      '<path d="M88 146H224L245 158H87Z" fill="#c2cdc1" stroke="#8b9f90"/>'+
      '<path d="M109 151H214" stroke="#f3f7f0" stroke-width="2"/>'+
      '<g class="screen" transform="translate(88 145) rotate(-'+angle+')"><rect x="0" y="-6" width="114" height="10" rx="4" fill="'+lid+'"/><path d="M8 -3H105" stroke="#d7e9dc" stroke-width="2"/></g>'+
      '<path d="M32 149V125L48 78L'+ex.toFixed(1)+' '+(ey-11).toFixed(1)+'" fill="none" stroke="'+arm+'" stroke-width="11" stroke-linecap="round" stroke-linejoin="round"/>'+
      '<circle cx="48" cy="78" r="8" fill="#f7f9f2" stroke="'+arm+'" stroke-width="4"/>'+
      '<circle cx="32" cy="126" r="7" fill="#f7f9f2" stroke="'+arm+'" stroke-width="4"/>'+
      '<path d="M'+(ex-5).toFixed(1)+' '+(ey-14).toFixed(1)+'v9l5 5m5-14v9l-5 5" stroke="#627a6b" fill="none" stroke-width="3"/>'+
      '<rect x="20" y="144" width="25" height="15" rx="3" fill="#a9b8a6"/>'+
      tiles+'<text x="249" y="70" font-size="8" fill="#7a8d7b" font-family="monospace">latent</text>'+
      '<path d="M108 145 A20 20 0 0 0 '+(88+20*Math.cos(rad)).toFixed(1)+' '+(145-20*Math.sin(rad)).toFixed(1)+'" fill="none" stroke="#b18a43" stroke-width="1.4"/>';
  }
  const stepButtons=document.querySelector('.step-buttons');
  stepButtons.innerHTML=Array.from({length:6},(_,m)=>'<button data-step="'+m+'" aria-pressed="'+(m===0)+'">'+(m===0?'真实起点':'预测 '+m)+'<span>'+(m===0?'z₀':'ẑ'+m)+'</span></button>').join('');
  document.querySelector('.state-track').innerHTML=Array.from({length:5},(_,m)=>'<div class="state-chip" data-state-step="'+(m+1)+'">s₀ <span style="opacity:.6">→ m'+(m+1)+'</span></div>').join('');
  function fullScore(i) { return (candidates[i].angles[5]-20)/60*(roll.task===0?1:-1); }
  function renderRollout() {
    const candidate=candidates[roll.candidate], angle=candidate.angles[roll.step], sign=roll.task===0?1:-1;
    scene($('real-scene'),20,false); scene($('imagined-scene'),angle,true);
    $('imagined-time').textContent=roll.step===0?'起点 z₀':'假想 A'+roll.step+' · m = '+roll.step;
    $('latent-badge').textContent=roll.step===0?'z₀ · 真实图像特征':'ẑ'+roll.step+' · 预测特征';
    $('angle-label').textContent=angle+'° · 示意';
    $('step-label').textContent=roll.step+' / 5';
    $('cumulative-score').textContent=fmt((angle-20)/60*sign,2);
    $('prev-step').disabled=roll.step===0; $('next-step').disabled=roll.step===5;
    $('roll-candidate').value=String(roll.candidate);
    document.querySelectorAll('[data-task]').forEach(b=>{const active=Number(b.dataset.task)===roll.task;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
    document.querySelectorAll('[data-step]').forEach(b=>{const s=Number(b.dataset.step);b.classList.toggle('current',s===roll.step);b.classList.toggle('past',s<roll.step);b.setAttribute('aria-pressed',String(s===roll.step));});
    document.querySelectorAll('[data-state-step]').forEach(c=>c.classList.toggle('current',Number(c.dataset.stateStep)===roll.step));
    $('state-caption').textContent=roll.step===0?'红色 s₀ 在每个子块都有一个位置。这里演示只传入一份当前状态时的复制分支。':'第 '+roll.step+' 子块：输入仍有 s₀，视觉输入为 '+(roll.step===1?'真实起点 z₀':'上一子块预测的 ẑ'+(roll.step-1))+'。已有动作与 latent 历史继续保留。';
    $('progress-bars').innerHTML=Array.from({length:5},(_,i)=>{
      const p=(candidate.angles[i+1]-candidate.angles[i])/60*sign;
      const height=i<roll.step?Math.min(34,Math.abs(p)*125):0;
      return '<div class="bar-col" title="子块 '+(i+1)+'：'+fmt(p,3)+'"><i class="'+(p<0?'negative':'')+'" style="height:'+height+'px"></i><b>'+(i+1)+'</b></div>';
    }).join('');
    const scores=candidates.map((_,i)=>fullScore(i)), best=scores.indexOf(Math.max(...scores));
    $('candidate-scores').innerHTML=candidates.map((c,i)=>'<button class="candidate '+(i===roll.candidate?'active ':'')+(i===best?'best':'')+'" data-candidate="'+i+'" aria-pressed="'+(i===roll.candidate)+'"><span>'+c.name+'</span><span>'+(i===best?'<span class="winner">最高</span> ':'')+fmt(scores[i],2)+'</span></button>').join('');
  }
  function stopRollout() { if(roll.timer!==null)clearInterval(roll.timer);roll.timer=null;$('play-rollout').textContent='▶ 播放';$('play-rollout').setAttribute('aria-label','播放预演'); }
  function setStep(m) { stopRollout();roll.step=Math.max(0,Math.min(5,m));renderRollout();announce('预演第 '+roll.step+' 步，真实机器人仍在起点。'); }
  function playRollout() {
    if(roll.timer!==null){stopRollout();return;}
    if(roll.step===5)roll.step=0;
    renderRollout();$('play-rollout').textContent='Ⅱ 暂停';$('play-rollout').setAttribute('aria-label','暂停预演');
    roll.timer=setInterval(()=>{roll.step++;renderRollout();if(roll.step>=5)stopRollout();},1400);
  }
  $('play-rollout').addEventListener('click',playRollout);
  $('prev-step').addEventListener('click',()=>setStep(roll.step-1));
  $('next-step').addEventListener('click',()=>setStep(roll.step+1));
  $('reset-rollout').addEventListener('click',()=>setStep(0));
  stepButtons.addEventListener('click',e=>{const b=e.target.closest('[data-step]');if(b)setStep(Number(b.dataset.step));});
  document.querySelectorAll('[data-task]').forEach(b=>b.addEventListener('click',()=>{roll.task=Number(b.dataset.task);renderRollout();}));
  $('roll-candidate').addEventListener('change',e=>{roll.candidate=Number(e.target.value);renderRollout();});
  $('candidate-scores').addEventListener('click',e=>{const b=e.target.closest('[data-candidate]');if(b){roll.candidate=Number(b.dataset.candidate);renderRollout();}});
  $('start-demo').addEventListener('click',()=>{setStep(0);$('rollout').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth'});playRollout();});
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stopRollout();});
  renderRollout();

  // Actual teaching calculation: LN → Q,K,V → mask → softmax → AV.
  const math=window.VerifierLabMath;
  const att={T:2,task:0,isolate:true,row:15,col:11,data:null};
  const colors={action:'#9c732b',state:'#bd5951',visual:'#527eaf',task:'#8962a2'};
  const names={action:'动作',state:'状态',visual:'视觉',task:'任务'};
  function tname(t) { return names[t.kind]+' '+t.label; }
  function shortName(t) { return (t.block+1)+'·'+t.label; }
  function renderAttention() {
    const d=att.data=math.calculate(att.T,att.task,att.isolate);
    const count=d.tokens.length;
    att.row=Math.min(att.row,count-1);att.col=Math.min(att.col,count-1);
    let markup='<thead><tr><th scope="col">Q ↓ / K →</th>';
    d.tokens.forEach(t=>{markup+='<th scope="col" style="color:'+colors[t.kind]+'">'+shortName(t)+'</th>';});
    markup+='</tr></thead><tbody>';
    d.tokens.forEach((t,i)=>{
      markup+='<tr class="'+(i===att.row?'selected-row ':'')+(i===8?'block-start':'')+'"><th scope="row"><button data-reader="'+i+'" class="'+(i===att.row?'selected':'')+'" aria-label="读取者：第'+(t.block+1)+'子块的'+tname(t)+'" style="'+(i===att.row?'':'color:'+colors[t.kind])+'">'+shortName(t)+'</button></th>';
      d.tokens.forEach((k,j)=>{
        const w=d.weights[i][j], v=Math.min(1,w*2.8), color='rgb('+Math.round(239-210*v)+','+Math.round(245-128*v)+','+Math.round(236-136*v)+')';
        const label=shortName(t)+' 读取 '+shortName(k)+'；'+(d.allowed[i][j]?'权重 '+fmt(w,4):'被屏蔽，权重 0');
        markup+='<td class="'+(j===8?'block-start-col':'')+'"><button data-cell="'+i+','+j+'" class="'+(!d.allowed[i][j]?'blocked ':'')+(i===att.row&&j===att.col?'selected-cell':'')+'" style="background:'+color+'" aria-label="'+label+'" title="'+label+'"></button></td>';
      });
      markup+='</tr>';
    });
    $('attention-matrix').innerHTML=markup+'</tbody>';
    $('mask-status').textContent=att.isolate?'论文掩码':'反事实：任务隔离关闭';
    const alternate=math.calculate(att.T,1-att.task,att.isolate);
    const difference=math.visualDifference(d,alternate);
    $('visual-difference').textContent=fmt(difference,6);
    $('isolation-result').classList.toggle('warn',!att.isolate);
    const taskRow=d.tokens[att.row].kind==='task';
    $('read-task').classList.toggle('active',taskRow);$('read-task').setAttribute('aria-pressed',String(taskRow));
    const visualRow=d.tokens[att.row].kind==='visual';
    $('read-visual').classList.toggle('active',visualRow);$('read-visual').setAttribute('aria-pressed',String(visualRow));
    renderCalculation();renderShapes();
  }
  function renderCalculation() {
    const d=att.data,i=att.row,j=att.col,t=d.tokens[i],k=d.tokens[j];
    const raw=d.raw[i][j],w=d.weights[i][j],allowed=d.allowed[i][j];
    $('selected-pair').textContent=tname(t)+' → '+tname(k);
    $('pair-blocks').textContent='子块 '+(t.block+1)+' → 子块 '+(k.block+1);
    $('q-label').textContent=shortName(t)+' 的 Q · 4 维';
    $('k-label').textContent=shortName(k)+' 的 K · 4 维';
    $('q-vector').textContent=vector(d.Q[i]);$('k-vector').textContent=vector(d.K[j]);
    const terms=d.Q[i].map((v,n)=>fmt(v,2)+'×('+fmt(d.K[j][n],2)+')').join(' + ');
    let reason='';
    if(!allowed)reason=k.block>t.block?'未来子块不可读':'任务 token 的列被隔离';
    $('dot-product').innerHTML='<span>① 点积并缩放</span><br>('+terms+') / √4<br>≈ <strong>'+fmt(raw,4)+'</strong><br><span>② 应用掩码 → ③ 行 softmax</span><br>'+(allowed?'保留该分数 → 权重 <strong>'+fmt(w,4)+'</strong>':reason+' → −∞ → 权重 <strong>0</strong>');
    $('weight-sum').textContent='权重之和 = '+fmt(d.weights[i].reduce((a,b)=>a+b,0),6);
    $('attention-weights').innerHTML=d.tokens.map((token,n)=>'<button class="weight-pill '+(!d.allowed[i][n]?'blocked ':'')+(n===j?'selected':'')+'" data-key="'+n+'" aria-label="查看 '+shortName(token)+' 的权重 '+fmt(d.weights[i][n],4)+'"><small style="color:'+colors[token.kind]+'">'+shortName(token)+'</small>'+fmt(d.weights[i][n],3)+'</button>').join('');
    $('weighted-value').textContent='Y['+shortName(t)+'] = Σⱼ αⱼVⱼ ≈ '+vector(d.out[i]);
    $('normalized-vector').textContent=vector(d.normalized[i]);
    $('wq-matrix').textContent=math.WQ.map(r=>vector(r)).join('\n');
    if(t.kind==='task'){
      $('attention-explanation').textContent=att.isolate?'任务行读到的是动作、状态、视觉及自身任务信息。上面得到一个 4 维注意力输出；源码会拼接所有头并经过残差、FFN 和后续层，最后才把任务位置读成 Δp̂。':'现在任务列也向其他 token 开放。这是用于比较的反事实；视觉行可能被任务向量影响。真实方法使用任务隔离掩码。';
    }else{
      $('attention-explanation').textContent=att.isolate?'当前是动态 token 的读取行。任务 token 的权重严格为 0，所以只切换任务 embedding 不会改变本示例的视觉注意力输出。动作和状态仍然可以影响它。':'取消任务隔离后，动态 token 可以读取任务向量。切换任务会改变本示例中的部分视觉输出；时间上的因果掩码仍然保留。';
    }
  }
  $('attention-matrix').addEventListener('click',e=>{
    const cell=e.target.closest('[data-cell]'),reader=e.target.closest('[data-reader]');
    if(cell){[att.row,att.col]=cell.dataset.cell.split(',').map(Number);renderAttention();}
    else if(reader){att.row=Number(reader.dataset.reader);renderAttention();}
  });
  $('attention-weights').addEventListener('click',e=>{const b=e.target.closest('[data-key]');if(b){att.col=Number(b.dataset.key);renderAttention();}});
  $('att-task').addEventListener('change',e=>{att.task=Number(e.target.value);renderAttention();});
  $('att-blocks').addEventListener('change',e=>{att.T=Number(e.target.value);att.row=(att.T-1)*8+att.row%8;att.col=(att.T-1)*8+att.col%8;renderAttention();});
  $('remove-isolation').addEventListener('change',e=>{att.isolate=!e.target.checked;renderAttention();});
  $('read-task').addEventListener('click',()=>{att.row=att.T*8-1;att.col=(att.T-1)*8+3;renderAttention();});
  $('read-visual').addEventListener('click',()=>{att.row=(att.T-1)*8+3;att.col=att.T*8-1;renderAttention();});
  function renderShapes() {
    const B=Number($('shape-batch').value),T=att.T,n=T*268,cache=$('shape-cache').checked,qLen=cache?268:n;
    $('shape-x').textContent='['+B+', '+qLen+', 1024]';
    $('shape-qkv').textContent=cache?'Q ['+B+',16,268,64]\nK/V ['+B+',16,'+n+',64]':'Q/K/V ['+B+',16,'+n+',64]';
    $('shape-scores').textContent='['+B+',16,'+qLen+','+n+']';
    $('shape-head').textContent='['+B+',16,'+qLen+',64]';
    $('shape-merged').textContent='['+B+','+qLen+',16×64] → ['+B+','+qLen+',1024]';
    $('shape-outputs').textContent=cache?'ẑ: ['+B+',256,768]    Δp̂: ['+B+',1]':'ẑ: ['+B+','+(T*256)+',768]    Δp̂: ['+B+','+T+',1]';
    $('shape-note').textContent='当前 T = '+T+'，累计 '+n+' tokens；每个头 dₕ = 1024 / 16 = 64。线性层打包输出为 ['+B+','+qLen+',3072]，再拆成 Q/K/V 和头维。'+(cache?'开启缓存后只为当前子块计算 Q；K/V 拼接历史缓存，因此分数矩阵可以是长方形。上方小矩阵仍展示完整上下文的掩码关系。':'普通 forward 同时处理已有的 T 个子块，因此 Q 与 K 的序列长度相同。')+' 这些是演示配置的形状推导，不是模型运行日志。';
  }
  $('shape-batch').addEventListener('change',renderShapes);$('shape-cache').addEventListener('change',renderShapes);
  renderAttention();

  const codeExamples={
    assemble:{title:'把四类 tokens 放进同一序列',file:'ac_predictor.py',range:'203-L213',explanation:'a 是每个动作的 token，s 是每个子块的一个状态 token，visual 是该子块的视觉特征，v 是任务 token。图示默认不使用可选的相机外参。flatten 将子块维与块内 token 维合成序列维；它不会混合 batch 中的候选。',lines:[
      'B, T, HW, D = visual.shape',
      'a, s = self._encode_action_state(actions, states)',
      'v = self._delta_progress_query(task_id, T, B, visual.device)',
      '',
      '# use_extrinsics=False 分支',
      'seq = torch.cat([a, s, visual, v], dim=2)',
      'return seq.flatten(1, 2)',
      '',
      '# [B,T,10,D] + [B,T,1,D] + [B,T,256,D] + [B,T,1,D]',
      '# -> [B,T,268,D] -> [B,T*268,D]'
    ]},
    qkv:{title:'任务 token 也生成 Q、K、V',file:'attention.py',range:'438-L451',explanation:'任务位置的 value_x 使用与其他 token 相同的 self.qkv 线性层；不是另建一套跨注意力。先投影到 3D，再拆成 3 × heads × head_dim，换轴得到 [3,B,heads,T,head_dim]（每块只有一个任务 token）。源码接下来给 Q、K 的时间分量施加 RoPE，再与其他 tokens 的 Q/K/V 拼回同一序列。',lines:[
      '# self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)',
      'value_qkv = (',
      '    self.qkv(value_x.flatten(1, 2))',
      '    .unflatten(-1, (3, self.num_heads, -1))',
      '    .permute(2, 0, 3, 1, 4)',
      ')',
      'value_q, value_k, value_v = (',
      '    value_qkv[0], value_qkv[1], value_qkv[2]',
      ')',
      '# 任务输入 [B,T,1024] -> 投影 [B,T,3072]',
      '# -> [3,B,16,T,64]'
    ]},
    mask:{title:'任务列先全部关闭，再开放自身',file:'attention.py',range:'39-L56',explanation:'矩阵行表示读取者，列表示被读取者。先开放当前和过去的子块，再关闭所有任务 token 的列。最后只恢复任务 token 对自己的读取。于是动态 token 无法从任何子块读取任务信息；当前任务 token 也不会读取过去的任务 token。布尔 True 表示允许读取。',lines:[
      '# 先建立块因果掩码',
      'for t1 in range(T):',
      '    for t2 in range(0, t1 + 1):',
      '        mask[t1*N_T:(t1+1)*N_T, t2*N_T:(t2+1)*N_T] = True',
      '',
      '# val_global 是所有任务 token 的位置',
      'mask[:, val_global] = False',
      'for t in range(T):',
      '    v = t * N_T + val_local',
      '    mask[v.unsqueeze(1), v.unsqueeze(0)] = True'
    ]},
    heads:{title:'同一主干，不同位置，两路读出',file:'ac_predictor.py',range:'268-L280',explanation:'任务头读取最后一个 token，视觉头读取中间 HW 个视觉 token。进度头的直接输入是已经通过注意力更新的任务向量，而不是输出后的 ẑ。当前 value_mlp 是 Linear(D,1)。因此两路输出共享主干，同时从不同位置获得结果。',lines:[
      'h = h.view(B, T, tokens_per_frame, D)',
      'value_tok = h[:, :, cond_tokens + HW, :]',
      '',
      'out = h[:, :, cond_tokens:cond_tokens + HW, :].flatten(1, 2)',
      'out = self.predictor_proj(self.predictor_norm(out))',
      '',
      'value_logits = self.value_mlp(value_tok)',
      'return out, value_logits',
      '',
      '# out: [B,T*256,768]',
      '# value_logits: [B,T,1]'
    ]},
    state:{title:'重复 s₀ 是一个输入分支',file:'ac_predictor.py',range:'215-L219',explanation:'当状态只有 [B,dₛ] 时，代码将它重复到每个子块。若已经是 [B,M,dₛ]，此函数原样返回。重复初始状态不会提供新的真实测量；将它用作起点条件需要与训练约定一致。论文未公开足够的训练与部署细节来确认实验中采用哪种调用方式。',lines:[
      '@staticmethod',
      'def _expand_states(states, steps):',
      '    if states is not None and states.dim() == 2:',
      '        states = states.unsqueeze(1).repeat(1, steps, 1)',
      '    return states',
      '',
      '# [B,14] -> [B,M,14]',
      '# [B,M,14] -> 保持原样'
    ]}
  };
  function showCode(name) {
    const c=codeExamples[name];$('code-title').textContent=c.title;
    $('code-link').textContent='verifier/'+c.file+':'+c.range.replace('-L','–')+' ↗';
    $('code-link').href=source(c.file,c.range);$('code-snippet').textContent=c.lines.join('\n');
    $('code-explanation').textContent=c.explanation;
    document.querySelectorAll('[data-code]').forEach(b=>{const active=b.dataset.code===name;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});
  }
  document.querySelectorAll('[data-code]').forEach(b=>b.addEventListener('click',()=>showCode(b.dataset.code)));
  showCode('assemble');
  if(window.renderMathInElement){
    renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],throwOnError:false,strict:'ignore'});
  }
  if(window.mermaid){
    mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',themeVariables:{fontFamily:'Segoe UI, Microsoft YaHei, sans-serif',fontSize:'13px',lineColor:'#8b9c89',edgeLabelBackground:'#fffefa',primaryTextColor:'#26312f',clusterBkg:'#fbfcf8',clusterBorder:'#c9d1c2'},flowchart:{useMaxWidth:true,htmlLabels:false,curve:'basis',nodeSpacing:22,rankSpacing:30}});
    mermaid.run({querySelector:'.mermaid'}).then(()=>document.documentElement.dataset.diagramReady='true').catch(err=>{document.documentElement.dataset.diagramReady='error';console.error(err);});
  }
  document.documentElement.dataset.labReady='true';
})();
