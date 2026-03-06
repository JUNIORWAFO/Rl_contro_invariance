"""
hull_classifiers.py
====================
Three learned hull classifiers + a unified PyTorch pretrainer.

All classifiers share the protocol:
    clf.process(record: EpisodeRecord) -> HullResult
    clf.full_reset()
    clf.is_valid_target  : bool
    clf.target           : Optional[np.ndarray]

Architectures
-------------
  TransformerHullClassifier — causal Transformer over trajectory steps
  MambaHullClassifier       — selective SSM (Mamba-style), O(T) complexity
  GNNHullClassifier         — dual GraphSAGE on trajectory + hull graphs
                              Dense hull graph (fully connected ≤ 128 nodes)

Pretrainer
----------
  HullClassifierPretrainer — unified pretrainer for all three.
    Three synthetic generators: GaussianCluster, LinearSystem, ConvexHull
    Dual loss: BCE(binary) + λ MSE(soft distance label)
    Fine-tune strategy: encoder frozen, head only.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from shared_types import EpisodeRecord, HullResult

try:
    from hull_monitors import PostEpisodeHullMonitor, _scan_episode, _first_true
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, Delaunay
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH DATA  (for GNN)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Graph:
    node_feat:  "torch.Tensor"   # (N, F_n)
    edge_index: "torch.Tensor"   # (2, E)
    edge_feat:  "torch.Tensor"   # (E, F_e)
    @property
    def N(self): return self.node_feat.shape[0]

@dataclass
class GraphBatch:
    node_feat:  "torch.Tensor"
    edge_index: "torch.Tensor"
    edge_feat:  "torch.Tensor"
    batch_idx:  "torch.Tensor"
    n_nodes:    List[int]
    batch_size: int

    def to(self, device):
        return GraphBatch(self.node_feat.to(device), self.edge_index.to(device),
                          self.edge_feat.to(device), self.batch_idx.to(device),
                          self.n_nodes, self.batch_size)

def collate_graphs(graphs: List[Graph]) -> GraphBatch:
    nfs, eis, efs, bis = [], [], [], []
    off, ns = 0, []
    for b, g in enumerate(graphs):
        N = g.N; nfs.append(g.node_feat)
        eis.append(g.edge_index + off)
        efs.append(g.edge_feat)
        bis.append(torch.full((N,), b, dtype=torch.long))
        ns.append(N); off += N
    return GraphBatch(torch.cat(nfs), torch.cat(eis, 1),
                      torch.cat(efs), torch.cat(bis), ns, len(graphs))

def global_mean_pool(x, bi, B):
    out = torch.zeros(B, x.size(-1), device=x.device)
    cnt = torch.zeros(B, 1, device=x.device)
    out.scatter_add_(0, bi.unsqueeze(1).expand_as(x), x)
    cnt.scatter_add_(0, bi.unsqueeze(1), torch.ones(x.size(0),1,device=x.device))
    return out / cnt.clamp(1.)


# ── trajectory + hull graph builders ─────────────────────────────────────────

class TrajGraphBuilder:
    """Nodes = states, directed temporal edges + skip edges."""
    def __init__(self, xDim, uDim, max_skip=2):
        self.xDim = xDim; self.uDim = uDim; self.ms = max_skip
        self.nd = xDim*2+2; self.ed = uDim+xDim+1

    def build(self, states: np.ndarray, actions: np.ndarray) -> Graph:
        T = len(actions)
        s  = states[:T].astype(np.float32)
        sn = states[1:T+1].astype(np.float32)
        a  = actions[:T].astype(np.float32)
        t  = np.arange(T, dtype=np.float32)
        pe = np.stack([np.sin(2*np.pi*t/max(T,1)),
                       np.cos(2*np.pi*t/max(T,1))], 1)
        vel = np.zeros_like(s); vel[1:] = s[1:]-s[:-1]
        nf  = np.concatenate([s, pe, vel], 1)
        ss,ds,es=[],[],[]
        for k in range(1, self.ms+1):
            for t_ in range(T-k):
                d=sn[t_]-s[t_]; n=np.linalg.norm(d,keepdims=True)
                e=np.concatenate([a[t_],d,n])
                ss+=[t_,t_+k]; ds+=[t_+k,t_]; es+=[e,np.concatenate([a[t_],-d,n])]
        if not ss: ss,ds,es=[0],[0],[np.zeros(self.ed,np.float32)]
        return Graph(torch.tensor(nf), torch.tensor([ss,ds],dtype=torch.long),
                     torch.tensor(np.array(es,np.float32)))


class HullGraphBuilder:
    """
    Dense hull graph: fully connected for M ≤ 128 (no edge skipped),
    k-NN with k=M//4 for larger hulls.
    Rationale: without a reliable vertex oracle, sparse graphs may disconnect
    true boundary regions; dense graphs let SAGE learn to downweight irrelevant edges.
    """
    def __init__(self, xDim, max_nodes=128):
        self.xDim=xDim; self.mn=max_nodes
        self.nd=xDim+2; self.ed=xDim+1

    def _dummy(self):
        return Graph(torch.zeros(1,self.nd), torch.zeros(2,1,dtype=torch.long),
                     torch.zeros(1,self.ed))

    def build(self, pts: np.ndarray) -> Graph:
        M = len(pts)
        if M == 0: return self._dummy()
        if M > self.mn: pts = pts[-self.mn:]; M = self.mn
        nf = np.concatenate([pts.astype(np.float32),
                              np.ones((M,1),np.float32),
                              np.ones((M,1),np.float32)], 1)
        if M <= 128:
            # Dense: all pairs
            ii = np.array([(i,j) for i in range(M) for j in range(M) if i!=j])
        else:
            # k-NN fallback
            k = max(1, M//4)
            d = np.linalg.norm(pts[:,None]-pts[None,:],axis=-1)
            np.fill_diagonal(d,np.inf)
            ii = np.array([(i,j) for i in range(M)
                            for j in np.argsort(d[i])[:k]])
        if len(ii)==0:
            return self._dummy()
        srcs, dsts = ii[:,0], ii[:,1]
        disp = pts[dsts]-pts[srcs]
        dist = np.linalg.norm(disp,axis=1,keepdims=True)
        ef   = np.concatenate([disp.astype(np.float32), dist.astype(np.float32)], 1)
        return Graph(torch.tensor(nf),
                     torch.tensor([srcs,dsts],dtype=torch.long),
                     torch.tensor(ef))


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED BASE CLASSIFIER
# ══════════════════════════════════════════════════════════════════════════════

if _TORCH:

    class _BaseClassifier:
        """Common oracle+training logic for all three classifiers."""

        is_valid_target: bool = False
        target: Optional[np.ndarray] = None

        def __init__(self, oracle_every=10, train_every=5,
                     batch_size=8, train_steps=20, pos_weight=5.,
                     threshold=0.5, device="cpu"):
            self._oracle_every = oracle_every
            self._train_every  = train_every
            self._batch_size   = batch_size
            self._train_steps  = train_steps
            self._pos_weight   = pos_weight
            self._threshold    = threshold
            self._device       = torch.device(device)
            self._ep_count     = 0
            self._label_count  = 0
            self._trained      = False
            self._buf: deque   = deque(maxlen=500)
            self._oracle       = None           # set in subclass __init__

        def _build_oracle(self, xDim, uDim, action_bounds, target):
            # Defer import to avoid circular
            from hull_monitors import PostEpisodeHullMonitor
            self._oracle       = PostEpisodeHullMonitor(target)
            self.target        = (np.asarray(target, np.float32)
                                  if target is not None else None)
            self.is_valid_target = target is not None

        def process(self, record: EpisodeRecord) -> HullResult:
            self._ep_count += 1
            run_oracle = (self._ep_count % self._oracle_every == 0
                          or not self._trained)
            if run_oracle:
                return self._oracle_step(record)
            return self._clf_step(record)

        def _oracle_step(self, record):
            result = self._oracle.process(record)
            if self._oracle.is_valid_target and not self.is_valid_target:
                self.target          = self._oracle.target
                self.is_valid_target = True
            labels = result.in_hull_mask.astype(np.float32)
            self._buf.append(self._make_sample(record, labels))
            self._label_count += 1
            if (self._label_count % self._train_every == 0
                    and len(self._buf) >= max(2, self._batch_size)):
                self._train()
                self._trained = True
            return result

        def _clf_step(self, record) -> HullResult:
            probs = self._predict(record)
            T     = record.T
            if len(probs) < T:
                probs = np.concatenate([probs, np.full(T-len(probs), .5)])
            probs    = probs[:T]
            mask     = probs >= self._threshold
            first    = _first_true(mask)
            return HullResult(mask, first, self.is_valid_target,
                              getattr(self, "name", "learned"))

        def _make_sample(self, record, labels):
            raise NotImplementedError

        def _predict(self, record) -> np.ndarray:
            raise NotImplementedError

        def _train(self):
            raise NotImplementedError

        def full_reset(self):
            self.target = None; self.is_valid_target = False
            self._ep_count = self._label_count = 0
            self._trained = False; self._buf.clear()
            if self._oracle: self._oracle.full_reset()


    # ══════════════════════════════════════════════════════════════════════════
    #  1. TRANSFORMER HULL CLASSIFIER
    # ══════════════════════════════════════════════════════════════════════════

    class _CausalAttention(nn.Module):
        def __init__(self, d, heads, drop):
            super().__init__()
            self.h=heads; self.dh=d//heads; self.scale=(d//heads)**-.5
            self.qkv=nn.Linear(d,3*d,bias=False); self.proj=nn.Linear(d,d)
            self.drop=nn.Dropout(drop)
        def forward(self, x):
            B,T,_=x.shape; H,Dh=self.h,self.dh
            q,k,v=self.qkv(x).split(H*Dh,dim=-1)
            q=q.view(B,T,H,Dh).transpose(1,2)
            k=k.view(B,T,H,Dh).transpose(1,2)
            v=v.view(B,T,H,Dh).transpose(1,2)
            att=(q@k.transpose(-2,-1))*self.scale
            mask=torch.triu(torch.ones(T,T,device=x.device),1).bool()
            att=att.masked_fill(mask,-1e9)
            att=self.drop(F.softmax(att,-1))
            return self.proj((att@v).transpose(1,2).reshape(B,T,H*Dh))

    class _TFBlock(nn.Module):
        def __init__(self, d, heads, ff, drop):
            super().__init__()
            self.ln1=nn.LayerNorm(d); self.attn=_CausalAttention(d,heads,drop)
            self.ln2=nn.LayerNorm(d)
            self.ff=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Linear(ff,d))
            self.drop=nn.Dropout(drop)
        def forward(self,x):
            x=x+self.drop(self.attn(self.ln1(x)))
            x=x+self.drop(self.ff(self.ln2(x))); return x

    class TransformerHullClassifier(_BaseClassifier):
        """Causal GPT-style Transformer over trajectory steps."""
        name = "transformer"

        def __init__(self, xDim, uDim, action_bounds=None, target=None,
                     d_model=64, n_heads=4, n_layers=3, ff=128, drop=0.1,
                     lr=3e-4, max_len=512, **kw):
            super().__init__(**kw)
            self._build_oracle(xDim, uDim, action_bounds, target)
            feat  = xDim + uDim + xDim   # state + action + delta
            self._net = nn.Sequential(
                nn.Linear(feat, d_model), nn.ReLU(),
                *[_TFBlock(d_model, n_heads, ff, drop) for _ in range(n_layers)],
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1),
            ).to(self._device)
            # Wrap as module that does embedding then blocks then head
            self._embed  = nn.Linear(feat, d_model).to(self._device)
            self._blocks = nn.ModuleList([_TFBlock(d_model,n_heads,ff,drop)
                                          for _ in range(n_layers)]).to(self._device)
            self._head   = nn.Sequential(nn.LayerNorm(d_model),
                                          nn.Linear(d_model,1)).to(self._device)
            self._opt    = optim.Adam(
                list(self._embed.parameters())+
                list(self._blocks.parameters())+
                list(self._head.parameters()), lr=lr)
            self._xDim=xDim; self._uDim=uDim; self._ml=max_len

        def _encode(self, record) -> np.ndarray:
            T=record.T; s=record.states[:T]; a=record.actions; sn=record.states[1:T+1]
            ds=sn-s; feat=np.concatenate([s,a,ds],1).astype(np.float32)
            return feat

        def _make_sample(self, record, labels):
            return (self._encode(record), labels)

        def _predict(self, record) -> np.ndarray:
            self._embed.eval(); self._head.eval()
            with torch.no_grad():
                feat=torch.tensor(self._encode(record)).unsqueeze(0).to(self._device)
                x=F.relu(self._embed(feat))
                for b in self._blocks: x=b(x)
                logit=self._head(x).squeeze(-1).squeeze(0)
            return torch.sigmoid(logit).cpu().numpy()

        def _train(self):
            for _ in range(self._train_steps):
                idxs=[np.random.randint(len(self._buf))
                      for _ in range(self._batch_size)]
                mb  =[self._buf[i] for i in idxs]
                loss_total=torch.tensor(0., device=self._device)
                for feat_np, lbl_np in mb:
                    T=min(len(lbl_np), self._ml)
                    feat=torch.tensor(feat_np[:T]).unsqueeze(0).to(self._device)
                    lbl =torch.tensor(lbl_np[:T]).to(self._device)
                    x=F.relu(self._embed(feat))
                    for b in self._blocks: x=b(x)
                    logit=self._head(x).squeeze(-1).squeeze(0)
                    prob =torch.sigmoid(logit)
                    w=torch.where(lbl>0.5,
                                  torch.full_like(lbl,self._pos_weight),
                                  torch.ones_like(lbl))
                    loss_total=loss_total+F.binary_cross_entropy(prob,lbl,weight=w)
                self._opt.zero_grad()
                (loss_total/len(mb)).backward()
                nn.utils.clip_grad_norm_(
                    list(self._embed.parameters())+
                    list(self._blocks.parameters())+
                    list(self._head.parameters()), 1.0)
                self._opt.step()

        def freeze_encoder(self):
            for p in self._embed.parameters(): p.requires_grad_(False)
            for b in self._blocks: 
                for p in b.parameters(): p.requires_grad_(False)

        def unfreeze_head(self):
            for p in self._head.parameters(): p.requires_grad_(True)


    # ══════════════════════════════════════════════════════════════════════════
    #  2. MAMBA SSM HULL CLASSIFIER
    # ══════════════════════════════════════════════════════════════════════════

    class _SelectiveSSM(nn.Module):
        """Minimal Mamba-style selective SSM block (pure PyTorch, no mamba lib)."""
        def __init__(self, d, d_state=16, d_conv=4, expand=2):
            super().__init__()
            di=d*expand
            self.in_proj  =nn.Linear(d,di*2,bias=False)
            self.conv1d   =nn.Conv1d(di,di,d_conv,padding=d_conv-1,groups=di)
            self.x_proj   =nn.Linear(di,d_state*2+1,bias=False)
            self.dt_proj  =nn.Linear(1,di)
            self.out_proj =nn.Linear(di,d,bias=False)
            self.A=nn.Parameter(torch.log(torch.arange(1,d_state+1,dtype=torch.float)
                                           .unsqueeze(0).expand(di,-1)))
            self.D=nn.Parameter(torch.ones(di))
            self.d=d; self.di=di; self.ds=d_state

        def forward(self, x):
            B,T,_=x.shape; di=self.di; ds=self.ds
            xz=self.in_proj(x)
            x_,z=xz[...,:di],xz[...,di:]
            x_=self.conv1d(x_.transpose(1,2))[...,:T].transpose(1,2)
            x_=F.silu(x_)
            z =F.silu(z)
            ssm=self.x_proj(x_)
            B_=ssm[...,:ds]; C=ssm[...,ds:2*ds]
            dt =F.softplus(self.dt_proj(ssm[...,2:]))
            A  =-torch.exp(self.A)
            dA =torch.exp(dt.unsqueeze(-1)*A.unsqueeze(0).unsqueeze(0))
            dB =dt.unsqueeze(-1)*B_.unsqueeze(-2)
            h  =torch.zeros(B,di,ds,device=x.device)
            ys=[]
            for t_ in range(T):
                h=dA[:,t_]*h+dB[:,t_]*x_[:,t_:t_+1,:]
                ys.append((h*C[:,t_:t_+1,:].unsqueeze(1)).sum(-1).sum(1))
            y=torch.stack(ys,1)+self.D*x_
            return self.out_proj(y*z)

    class _MambaBlock(nn.Module):
        def __init__(self, d, d_state, d_conv, expand, drop):
            super().__init__()
            self.norm=nn.LayerNorm(d)
            self.ssm =_SelectiveSSM(d,d_state,d_conv,expand)
            self.drop=nn.Dropout(drop)
        def forward(self,x): return x+self.drop(self.ssm(self.norm(x)))

    class MambaHullClassifier(_BaseClassifier):
        """Selective SSM (Mamba) classifier — O(T) complexity."""
        name = "mamba"

        def __init__(self, xDim, uDim, action_bounds=None, target=None,
                     d_model=64, n_layers=3, d_state=16, d_conv=4, expand=2,
                     drop=0.1, lr=3e-4, **kw):
            super().__init__(**kw)
            self._build_oracle(xDim, uDim, action_bounds, target)
            feat=xDim+uDim+xDim
            self._embed=nn.Linear(feat,d_model).to(self._device)
            self._blocks=nn.ModuleList([_MambaBlock(d_model,d_state,d_conv,expand,drop)
                                         for _ in range(n_layers)]).to(self._device)
            self._head=nn.Sequential(nn.LayerNorm(d_model),
                                      nn.Linear(d_model,1)).to(self._device)
            self._opt=optim.Adam(
                list(self._embed.parameters())+
                list(self._blocks.parameters())+
                list(self._head.parameters()), lr=lr)
            self._xDim=xDim; self._uDim=uDim

        def _encode(self, record):
            T=record.T; s=record.states[:T]; a=record.actions
            ds=record.states[1:T+1]-s
            return np.concatenate([s,a,ds],1).astype(np.float32)

        def _make_sample(self,record,labels): return (self._encode(record), labels)

        def _predict(self,record):
            self._embed.eval(); self._head.eval()
            with torch.no_grad():
                f=torch.tensor(self._encode(record)).unsqueeze(0).to(self._device)
                x=F.relu(self._embed(f))
                for b in self._blocks: x=b(x)
                logit=self._head(x).squeeze(-1).squeeze(0)
            return torch.sigmoid(logit).cpu().numpy()

        def _train(self):
            for _ in range(self._train_steps):
                idxs=[np.random.randint(len(self._buf)) for _ in range(self._batch_size)]
                mb  =[self._buf[i] for i in idxs]
                loss_total=torch.tensor(0.,device=self._device)
                for feat_np,lbl_np in mb:
                    T=len(lbl_np)
                    f  =torch.tensor(feat_np[:T]).unsqueeze(0).to(self._device)
                    lbl=torch.tensor(lbl_np[:T]).to(self._device)
                    x=F.relu(self._embed(f))
                    for b in self._blocks: x=b(x)
                    prob=torch.sigmoid(self._head(x).squeeze(-1).squeeze(0))
                    w=torch.where(lbl>0.5,
                                  torch.full_like(lbl,self._pos_weight),
                                  torch.ones_like(lbl))
                    loss_total=loss_total+F.binary_cross_entropy(prob,lbl,weight=w)
                self._opt.zero_grad()
                (loss_total/len(mb)).backward()
                nn.utils.clip_grad_norm_(
                    list(self._embed.parameters())+
                    list(self._blocks.parameters())+
                    list(self._head.parameters()), 1.0)
                self._opt.step()

        def freeze_encoder(self):
            for p in self._embed.parameters(): p.requires_grad_(False)
            for b in self._blocks:
                for p in b.parameters(): p.requires_grad_(False)


    # ══════════════════════════════════════════════════════════════════════════
    #  3. GNN HULL CLASSIFIER  (dense hull graph)
    # ══════════════════════════════════════════════════════════════════════════

    class _SAGELayer(nn.Module):
        def __init__(self, d_in, d_out, d_edge, drop=0.1):
            super().__init__()
            self.em=nn.Sequential(nn.Linear(d_in+d_edge,d_out),nn.ReLU())
            self.up=nn.Sequential(nn.Linear(d_in+d_out,d_out),nn.ReLU())
            self.norm=nn.LayerNorm(d_out); self.drop=nn.Dropout(drop)
            self.res=(nn.Linear(d_in,d_out,bias=False) if d_in!=d_out else nn.Identity())
        def forward(self,x,ei,ef):
            src,dst=ei[0],ei[1]; N=x.size(0)
            msg=self.em(torch.cat([x[src],ef],1))
            agg=torch.zeros(N,msg.size(-1),device=x.device)
            cnt=torch.zeros(N,1,device=x.device)
            agg.scatter_add_(0,dst.unsqueeze(1).expand_as(msg),msg)
            cnt.scatter_add_(0,dst.unsqueeze(1),torch.ones(msg.size(0),1,device=x.device))
            agg=agg/cnt.clamp(1.)
            out=self.up(torch.cat([x,agg],1))
            return self.norm(self.drop(out)+self.res(x))

    class _SAGEEncoder(nn.Module):
        def __init__(self, nd, ed, d, n_layers, drop=0.1):
            super().__init__()
            self.ip=nn.Linear(nd,d); self.ep=nn.Linear(ed,d)
            self.layers=nn.ModuleList([_SAGELayer(d,d,d,drop) for _ in range(n_layers)])
            self.gp=nn.Sequential(nn.Linear(d*2,d),nn.ReLU(),nn.LayerNorm(d))
        def forward(self, g: GraphBatch):
            x=F.relu(self.ip(g.node_feat)); ef=F.relu(self.ep(g.edge_feat))
            for l in self.layers: x=l(x,g.edge_index,ef)
            B=g.batch_size; bi=g.batch_idx
            gm=global_mean_pool(x,bi,B)
            gx=torch.zeros_like(gm).scatter_reduce_(
                0,bi.unsqueeze(1).expand_as(x),x,reduce="amax",include_self=True)
            return x, self.gp(torch.cat([gm,gx],1))

    class GNNHullClassifier(_BaseClassifier):
        """Dual GraphSAGE on trajectory graph + dense hull graph."""
        name = "gnn"

        def __init__(self, xDim, uDim, action_bounds=None, target=None,
                     d_model=64, traj_layers=3, hull_layers=2,
                     hidden=64, drop=0.1, lr=3e-4, **kw):
            super().__init__(**kw)
            self._build_oracle(xDim, uDim, action_bounds, target)
            self._tb  = TrajGraphBuilder(xDim, uDim)
            self._hb  = HullGraphBuilder(xDim)
            self._D   = d_model

            self._traj_enc = _SAGEEncoder(self._tb.nd, self._tb.ed,
                                           d_model, traj_layers, drop).to(self._device)
            self._hull_enc = _SAGEEncoder(self._hb.nd, self._hb.ed,
                                           d_model, hull_layers, drop).to(self._device)
            self._head = nn.Sequential(
                nn.LayerNorm(d_model*2), nn.Linear(d_model*2, hidden),
                nn.GELU(), nn.Dropout(drop), nn.Linear(hidden,1),
            ).to(self._device)
            self._opt = optim.Adam(
                list(self._traj_enc.parameters())+
                list(self._hull_enc.parameters())+
                list(self._head.parameters()), lr=lr)

        def _hull_graph(self) -> Graph:
            if self.target is None:
                return Graph(torch.zeros(1,self._hb.nd),
                             torch.zeros(2,1,dtype=torch.long),
                             torch.zeros(1,self._hb.ed))
            return self._hb.build(self.target)

        def _make_sample(self, record, labels):
            tg = self._tb.build(record.states, record.actions)
            hg = self._hull_graph()
            return (tg, hg, labels)

        def _forward_batch(self, traj_gs, hull_gs):
            tb = collate_graphs(traj_gs).to(self._device)
            hb = collate_graphs(hull_gs).to(self._device)
            tne, _   = self._traj_enc(tb)
            _,  hg_e = self._hull_enc(hb)
            hull_per_node = hg_e[tb.batch_idx]
            feat = torch.cat([tne, hull_per_node], 1)
            return torch.sigmoid(self._head(feat).squeeze(-1)), tb.n_nodes

        def _predict(self, record) -> np.ndarray:
            self._traj_enc.eval(); self._hull_enc.eval(); self._head.eval()
            with torch.no_grad():
                tg = self._tb.build(record.states, record.actions)
                hg = self._hull_graph()
                p, nn_ = self._forward_batch([tg], [hg])
            return p.cpu().numpy()[:record.T]

        def _train(self):
            self._traj_enc.train(); self._hull_enc.train(); self._head.train()
            for _ in range(self._train_steps):
                idxs = [np.random.randint(len(self._buf))
                        for _ in range(self._batch_size)]
                mb   = [self._buf[i] for i in idxs]
                tgs  = [s[0] for s in mb]
                hgs  = [s[1] for s in mb]
                lbls = [s[2] for s in mb]
                probs, ns = self._forward_batch(tgs, hgs)
                lf, wf = [], []
                for l, n in zip(lbls, ns):
                    L = min(len(l), n)
                    lf.extend(l[:L].tolist())
                    wf.extend([self._pos_weight if v>0.5 else 1. for v in l[:L]])
                    if n > L: lf.extend([0.]*( n-L)); wf.extend([0.]*(n-L))
                lbl_t = torch.tensor(lf[:probs.size(0)], device=self._device)
                w_t   = torch.tensor(wf[:probs.size(0)], device=self._device)
                loss  = (F.binary_cross_entropy(probs, lbl_t, reduction="none")*w_t).mean()
                self._opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self._traj_enc.parameters())+
                    list(self._hull_enc.parameters())+
                    list(self._head.parameters()), 1.0)
                self._opt.step()

        def freeze_encoder(self):
            for p in self._traj_enc.parameters(): p.requires_grad_(False)
            for p in self._hull_enc.parameters(): p.requires_grad_(False)

        def unfreeze_head(self):
            for p in self._head.parameters(): p.requires_grad_(True)


    # ══════════════════════════════════════════════════════════════════════════
    #  UNIFIED PRETRAINER
    # ══════════════════════════════════════════════════════════════════════════

    class _SyntheticDataset:
        """Three generators: Gaussian, LinearSystem, ConvexHull."""
        def __init__(self, xDim, uDim):
            self.xDim=xDim; self.uDim=uDim

        def _gaussian(self):
            xD,uD=self.xDim,self.uDim
            T=np.random.randint(10,80)
            mu=np.random.randn(xD).astype(np.float32)*2.
            A=np.random.randn(xD,xD).astype(np.float32)
            S=A.T@A+np.eye(xD,dtype=np.float32)*0.1; Si=np.linalg.inv(S)
            r2=float(2.*xD)
            hp=np.random.multivariate_normal(mu,S,30).astype(np.float32)
            d2=np.einsum("ti,ij,tj->t",hp-mu,Si,hp-mu)
            hp=hp[d2<=r2] if (d2<=r2).any() else hp[:1]
            walk=[mu+np.random.randn(xD).astype(np.float32)*3.]
            for _ in range(T-1):
                walk.append(walk[-1]+np.random.randn(xD).astype(np.float32)*0.5)
            traj=np.array(walk,np.float32)
            acts=np.random.randn(T,uD).astype(np.float32)*0.3
            m2=np.einsum("ti,ij,tj->t",traj-mu,Si,traj-mu)
            in_m=(m2<=r2); r=np.sqrt(r2)
            dv=np.clip((r-np.sqrt(m2))/r,0.,1.).astype(np.float32)*in_m
            states=np.vstack([traj,traj[-1:]])
            return states,acts,hp,in_m.astype(np.float32),dv

        def _linear(self):
            xD,uD=self.xDim,self.uDim
            T=np.random.randint(15,80)
            A=np.random.randn(xD,xD).astype(np.float32)
            r=np.abs(np.linalg.eigvals(A)).max()
            if r>1e-8: A*=0.9/r
            B=np.random.randn(xD,uD).astype(np.float32)*0.5
            x=np.random.randn(xD).astype(np.float32)
            rxs=[x.copy()]
            for _ in range(200):
                x=A@x+B@np.random.uniform(-1,1,uD).astype(np.float32)
                rxs.append(x.copy())
            hp=np.array(rxs[::4],np.float32)
            x=np.random.randn(xD).astype(np.float32)*1.5
            xs,us=[x.copy()],[]
            for _ in range(T):
                u=np.random.uniform(-1,1,uD).astype(np.float32)
                x=A@x+B@u; xs.append(x.copy()); us.append(u.copy())
            traj=np.array(xs[:-1],np.float32); acs=np.array(us,np.float32)
            lo,hi=hp.min(0),hp.max(0)
            in_m=np.all((traj>=lo)&(traj<=hi),1)
            ctr=hp.mean(0); d=np.linalg.norm(traj-ctr,1)
            dm=np.linalg.norm(hp-ctr,1).max()+1e-8
            dv=np.clip(np.where(in_m,1.-d/dm,0.),0.,1.).astype(np.float32)
            return np.array(xs,np.float32),acs,hp,in_m.astype(np.float32),dv

        def _convex(self):
            if not _SCIPY: return self._gaussian()
            xD,uD=self.xDim,self.uDim
            M=np.random.randint(8,40); T=np.random.randint(10,80)
            hp=np.random.randn(M,xD).astype(np.float32)*3.
            try: hull=ConvexHull(hp)
            except: return self._gaussian()
            hp=hp[hull.vertices]
            states=np.random.uniform(hp.min(0)*1.5,hp.max(0)*1.5,(T,xD)).astype(np.float32)
            try: in_m=Delaunay(hp).find_simplex(states)>=0
            except: in_m=np.zeros(T,bool)
            ctr=hp.mean(0); d=np.linalg.norm(states-ctr,1)
            dm=np.linalg.norm(hp-ctr,1).max()+1e-8
            dv=np.clip(np.where(in_m,1.-d/dm,0.),0.,1.).astype(np.float32)
            acts=np.random.randn(T,uD).astype(np.float32)*0.3
            return np.vstack([states,states[-1:]]),acts,hp,in_m.astype(np.float32),dv

        def sample(self):
            choice=np.random.choice([0,1,2],p=[0.4,0.35,0.25])
            return [self._gaussian,self._linear,self._convex][choice]()


    class HullClassifierPretrainer:
        """
        Unified pretrainer for Transformer, Mamba, and GNN classifiers.

        Strategy: encoder frozen during pretraining; only head trains.
        This means pretrained weights serve as a warm-start for the encoder,
        and fine-tuning on real env data only adjusts the lightweight head.

        Loss: BCE(binary, pos_weight=5) + λ·MSE(soft distance label)
        """

        def __init__(self, classifier, xDim: int, uDim: int,
                     lr: float = 3e-4, pos_weight: float = 5.,
                     lambda_dist: float = 0.5, batch_size: int = 16,
                     device: str = "cpu"):
            self.clf        = classifier
            self.xDim       = xDim; self.uDim = uDim
            self.pos_weight = pos_weight; self.lam = lambda_dist
            self.batch_size = batch_size
            self.device     = torch.device(device)
            self.dataset    = _SyntheticDataset(xDim, uDim)
            self._tb        = TrajGraphBuilder(xDim, uDim)
            self._hb        = HullGraphBuilder(xDim)

            # Freeze encoder, collect head params
            classifier.freeze_encoder()
            self.head_params = list(classifier._head.parameters())
            self.opt = optim.AdamW(self.head_params, lr=lr, weight_decay=1e-4)

        def _loss(self, prob, lbl, dist, mask):
            w   = torch.where(lbl>0.5,
                              torch.full_like(lbl,self.pos_weight),
                              torch.ones_like(lbl))
            lc  = (F.binary_cross_entropy(prob, lbl, reduction="none")*w*mask
                   ).sum()/(mask.sum()+1e-8)
            dm  = (dist>0.)*mask
            ld  = ((prob-dist)**2*dm).sum()/(dm.sum()+1e-8)
            return lc+self.lam*ld, float(lc), float(ld)

        def _step_gnn(self, samples):
            tgs = [self._tb.build(s[0],s[1]) for s in samples]
            hgs = [self._hb.build(s[2]) for s in samples]
            probs, ns = self.clf._forward_batch(tgs, hgs)
            lf,df,mf=[],[],[]
            for s,n in zip(samples,ns):
                L=min(len(s[3]),n)
                lf.extend(s[3][:L].tolist()); df.extend(s[4][:L].tolist())
                mf.extend([1.]*L)
                if n>L: lf.extend([0.]*(n-L)); df.extend([0.]*(n-L)); mf.extend([0.]*(n-L))
            N=probs.size(0)
            lbl_t=torch.tensor(lf[:N],device=self.device)
            dis_t=torch.tensor(df[:N],device=self.device)
            msk_t=torch.tensor(mf[:N],device=self.device)
            return self._loss(probs,lbl_t,dis_t,msk_t)

        def _step_seq(self, samples):
            loss_total=torch.tensor(0.,device=self.device)
            for states,acts,hp,lbl,dist in samples:
                rec=EpisodeRecord(states,acts,np.zeros(len(acts)))
                feat=torch.tensor(self.clf._encode(rec)).unsqueeze(0).to(self.device)
                x=F.relu(self.clf._embed(feat))
                for b in self.clf._blocks: x=b(x)
                T=len(lbl)
                prob=torch.sigmoid(self.clf._head(x).squeeze(-1).squeeze(0))[:T]
                lbl_t=torch.tensor(lbl,device=self.device)
                dis_t=torch.tensor(dist,device=self.device)
                msk_t=torch.ones(T,device=self.device)
                l,_,__=self._loss(prob,lbl_t,dis_t,msk_t)
                loss_total=loss_total+l
            return loss_total/len(samples),0.,0.

        def pretrain(self, n_epochs: int = 20, spe: int = 200,
                      save_path: Optional[str] = None) -> dict:
            clf=self.clf; is_gnn=isinstance(clf,GNNHullClassifier)
            history={"loss":[],"acc":[]}
            print(f"\n{'━'*50}")
            print(f"  Pretraining {clf.name}  epochs={n_epochs}  spe={spe}")
            print(f"  Strategy: encoder frozen, head-only")
            print(f"{'━'*50}")
            print(f"  {'Ep':>4} | {'loss':>8} | {'acc':>7}")

            for ep in range(n_epochs):
                ep_loss=0.; ep_acc=0.; steps=0
                if is_gnn:
                    clf._traj_enc.train(); clf._hull_enc.train()
                clf._head.train()

                for i in range(0,spe,self.batch_size):
                    bs=min(self.batch_size,spe-i)
                    samples=[self.dataset.sample() for _ in range(bs)]
                    if is_gnn:
                        loss,_,__=self._step_gnn(samples)
                    else:
                        loss,_,__=self._step_seq(samples)
                    self.opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(self.head_params,1.0)
                    self.opt.step()
                    ep_loss+=loss.item(); steps+=1

                history["loss"].append(ep_loss/max(steps,1))
                print(f"  {ep:>4} | {history['loss'][-1]:>8.4f}")

            if save_path:
                torch.save(clf._head.state_dict(), save_path)
                print(f"  Head saved → {save_path}")
            return history

        @staticmethod
        def load_head(clf, path: str):
            clf._head.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"  Head loaded ← {path}")
