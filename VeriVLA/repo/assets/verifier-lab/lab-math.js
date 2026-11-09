/* Deterministic teaching vectors, not pretrained model activations. */
(function (root) {
  'use strict';
  const WQ = [[.7,.2,-.1,.3],[-.2,.8,.3,.1],[.4,-.1,.6,.2],[.1,.3,-.2,.9]];
  const WK = [[.6,-.1,.3,.2],[.2,.7,-.2,.1],[-.1,.4,.8,.2],[.3,.1,.2,.6]];
  const WV = [[.8,.2,.1,-.1],[.1,.6,-.2,.3],[.2,-.1,.7,.2],[-.2,.3,.1,.9]];
  const bases = [[.8,.2,-.4,.1],[.1,.9,.3,-.5],[.6,-.2,.4,.8],[.9,.3,.2,-.4],[-.4,.8,.1,.6],[.2,-.4,.9,.3],[.4,.1,-.5,.9]];
  const taskVectors = [[.8,-.3,.6,-.5],[-.5,.9,-.4,.6]];
  const kinds = ['action','action','state','visual','visual','visual','visual','task'];
  const labels = ['a₁','a₂','s₀','z₁','z₂','z₃','z₄','q'];
  function dot(a,b) { return a.reduce((s,x,i)=>s+x*b[i],0); }
  function linear(x,w) { return w[0].map((_,j)=>x.reduce((s,v,i)=>s+v*w[i][j],0)); }
  function norm(x) {
    const mean=x.reduce((a,b)=>a+b,0)/x.length;
    const variance=x.reduce((s,v)=>s+(v-mean)**2,0)/x.length;
    return x.map(v=>(v-mean)/Math.sqrt(variance+1e-5));
  }
  function tokens(T, task) {
    return Array.from({length:T*8},(_,i)=>{
      const m=Math.floor(i/8), p=i%8;
      let x=p===7 ? taskVectors[task].slice() : bases[p].slice();
      if(p>=3 && p<=6) x=x.map((v,j)=>v+m*.12*Math.sin((p+1)*(j+1)));
      return {index:i,block:m,local:p,kind:kinds[p],label:labels[p],x};
    });
  }
  function permitted(i,j,isolate) {
    if(Math.floor(j/8)>Math.floor(i/8)) return false;
    return !isolate || j%8!==7 || i===j;
  }
  function calculate(T=2,task=0,isolate=true) {
    const ts=tokens(T,task), normalized=ts.map(t=>norm(t.x));
    const Q=normalized.map(x=>linear(x,WQ));
    const K=normalized.map(x=>linear(x,WK));
    const V=normalized.map(x=>linear(x,WV));
    const raw=Q.map(q=>K.map(k=>dot(q,k)/2));
    const allowed=raw.map((r,i)=>r.map((_,j)=>permitted(i,j,isolate)));
    const weights=raw.map((r,i)=>{
      const valid=r.filter((_,j)=>allowed[i][j]), max=Math.max(...valid);
      const e=r.map((s,j)=>allowed[i][j]?Math.exp(s-max):0);
      const sum=e.reduce((a,b)=>a+b,0);
      return e.map(x=>x/sum);
    });
    const out=weights.map(w=>Array.from({length:4},(_,d)=>w.reduce((s,p,j)=>s+p*V[j][d],0)));
    return {tokens:ts,normalized,Q,K,V,raw,allowed,weights,out};
  }
  function visualDifference(a,b) {
    let max=0;
    a.tokens.forEach((t,i)=>{ if(t.kind==='visual') a.out[i].forEach((v,j)=>max=Math.max(max,Math.abs(v-b.out[i][j]))); });
    return max;
  }
  root.VerifierLabMath={calculate,visualDifference,permitted,dot,linear,norm,WQ,WK,WV};
})(typeof window!=='undefined'?window:globalThis);
