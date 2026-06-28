# ================================================================
# SUBIT-INK × Vesuvius Challenge — Kaggle Version (FINAL)
# Копіюй кожну секцію в окрему клітинку Kaggle Notebook
# ================================================================


# ════════════════════════════════════════════════════════════════
# КЛІТИНКА 1 — Середовище + автопошук даних
# ════════════════════════════════════════════════════════════════
import os, torch, numpy as np
from pathlib import Path

print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}  '
              f'({torch.cuda.get_device_properties(i).total_memory/1e9:.1f} GB)')
else:
    print('⚠️  GPU не знайдено! Settings → Accelerator → GPU T4 x2')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# Автопошук шляхів
print('\nСтруктура /kaggle/input/:')
for root, dirs, files in os.walk('/kaggle/input'):
    level = root.replace('/kaggle/input', '').count(os.sep)
    if level > 3: continue
    indent = '  ' * level
    print(f'{indent}{Path(root).name}/')
    for f in sorted(files)[:3]:
        print(f'{indent}  {f}')
    if len(files) > 3:
        print(f'{indent}  ... ще {len(files)-3} файлів')

# Знаходимо inklabels.png і surface_volume автоматично
INK_PATHS = list(Path('/kaggle/input').rglob('inklabels.png'))
TIF_DIRS  = list(set(p.parent for p in Path('/kaggle/input').rglob('*.tif')))

print(f'\nЗнайдено inklabels.png: {INK_PATHS}')
print(f'Знайдено папок з .tif:  {TIF_DIRS}')

# Беремо перший знайдений фрагмент
assert INK_PATHS, "inklabels.png не знайдено! Додай датасет: Add Data → vesuvius-challenge-ink-detection"
INK_PATH = INK_PATHS[0]
VOL_DIR  = sorted(TIF_DIRS, key=lambda p: len(list(p.glob('*.tif'))), reverse=True)[0]
tif_files = sorted(VOL_DIR.glob('*.tif'))

print(f'\n✓ INK_PATH: {INK_PATH}')
print(f'✓ VOL_DIR:  {VOL_DIR}')
print(f'✓ Шарів:    {len(tif_files)}')


# ════════════════════════════════════════════════════════════════
# КЛІТИНКА 2 — Завантаження даних
# ════════════════════════════════════════════════════════════════
import tifffile
from PIL import Image

LAYERS    = 65
CROP_SIZE = 2048

# Ink mask
ink_full = (np.array(Image.open(str(INK_PATH)).convert('L')) > 128).astype(np.float32)
H_ink, W_ink = ink_full.shape
print(f'ink_full: {ink_full.shape}  coverage={ink_full.mean()*100:.1f}%')

# Центр тексту
iy, ix = np.where(ink_full > 0)
cy_ink, cx_ink = int(iy.mean()), int(ix.mean())
print(f'Центр тексту: y={cy_ink}, x={cx_ink}')

# Реальний розмір .tif
sample = tifffile.imread(str(tif_files[0]))
H_tif, W_tif = sample.shape[-2], sample.shape[-1]
print(f'Розмір .tif: {sample.shape}')

# Масштабуємо координати ink → tif
scale_y = H_tif / H_ink
scale_x = W_tif / W_ink
cy_tif  = int(cy_ink * scale_y)
cx_tif  = int(cx_ink * scale_x)

# Crop координати
hc = CROP_SIZE // 2
y1 = max(0, min(cy_tif - hc, H_tif - CROP_SIZE))
x1 = max(0, min(cx_tif - hc, W_tif - CROP_SIZE))
y2, x2 = y1 + CROP_SIZE, x1 + CROP_SIZE
print(f'Crop tif: y={y1}:{y2}, x={x1}:{x2}')

y1_ink = int(y1 / scale_y); y2_ink = int(y2 / scale_y)
x1_ink = int(x1 / scale_x); x2_ink = int(x2 / scale_x)

# Завантажуємо шари
n_layers = min(LAYERS, len(tif_files))
print(f'\nЗавантажуємо {n_layers} шарів...')
layers = []
for i, f in enumerate(tif_files[:n_layers]):
    raw = tifffile.imread(str(f))
    # Нормалізуємо форму: завжди 2D
    if raw.ndim == 3: raw = raw[0]
    layer = raw[y1:y2, x1:x2].astype(np.float32)
    layers.append(layer)
    if (i+1) % 10 == 0:
        print(f'  {i+1}/{n_layers}  shape={layer.shape}')

volume = np.stack(layers)  # [D, H, W]
v_min, v_max = float(volume.min()), float(volume.max())
volume = (volume - v_min) / (v_max - v_min + 1e-8)

# Ink для crop зони
ink_crop = ink_full[y1_ink:y2_ink, x1_ink:x2_ink]
ink = np.array(Image.fromarray(ink_crop).resize(
    (volume.shape[2], volume.shape[1]), Image.NEAREST))

print(f'\nVolume: {volume.shape}  RAM={volume.nbytes/1e9:.2f} GB')
print(f'Ink:    {ink.shape}  coverage={ink.mean()*100:.1f}%')

# Попередній перегляд
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.patch.set_facecolor('#111')
for ax in axes: ax.set_facecolor('#111'); ax.axis('off')
mid = volume.shape[0] // 2
axes[0].imshow(volume[mid-10], cmap='bone'); axes[0].set_title(f'Шар {mid-10}', color='white')
axes[1].imshow(volume[mid],    cmap='bone'); axes[1].set_title(f'Шар {mid}',    color='white')
axes[2].imshow(ink,            cmap='gray'); axes[2].set_title('Ink GT',         color='white')
plt.tight_layout(); plt.show()
print('✓ Дані завантажено')


# ════════════════════════════════════════════════════════════════
# КЛІТИНКА 3 — Модель + всі функції
# ════════════════════════════════════════════════════════════════
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from sklearn.metrics import f1_score, precision_score, recall_score
import time

class DC3d(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv3d(i, o, 3, padding=1, bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True),
            nn.Conv3d(o, o, 3, padding=1, bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)

class SUBITNet(nn.Module):
    def __init__(self, base=32, dual=True):
        super().__init__()
        self.e1   = DC3d(1, base)
        self.e2   = nn.Sequential(nn.MaxPool3d((1,2,2)), DC3d(base,   base*2))
        self.e3   = nn.Sequential(nn.MaxPool3d((1,2,2)), DC3d(base*2, base*4))
        self.bot  = DC3d(base*4, base*8)
        self.d3   = DC3d(base*8+base*4, base*4)
        self.d2   = DC3d(base*4+base*2, base*2)
        self.d1   = DC3d(base*2+base,   base)
        self.attn = nn.Sequential(nn.Conv3d(base, 1, 1), nn.Softmax(dim=2))
        self.dual = dual
        if dual: self.hy = nn.Conv2d(base, 1, 1); self.hn = nn.Conv2d(base, 1, 1)
        else:    self.hb = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); b = self.bot(e3)
        d3 = self.d3(torch.cat([F.interpolate(b,  size=e3.shape[2:], mode='trilinear', align_corners=False), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode='trilinear', align_corners=False), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode='trilinear', align_corners=False), e1], 1))
        out = (d1 * self.attn(d1)).sum(dim=2)
        if self.dual: return self.hy(out), self.hn(out)
        return self.hb(out)

def make_belnap_labels(ink_mask, dil=4):
    ink = ink_mask.astype(bool)
    border = binary_dilation(ink, iterations=dil) & ~ink
    yang = np.zeros_like(ink_mask, np.float32)
    yin  = np.zeros_like(ink_mask, np.float32)
    yang[ink] = 1.0
    yang[border] = 1.0; yin[border] = 1.0
    yin[~ink & ~border] = 1.0
    return yang, yin

def train_model(model, tl, vl, epochs, dual, pw):
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pw_t  = torch.tensor([pw]).to(DEVICE)
    model.to(DEVICE)
    th=[]; vh=[]; best_val=float('inf'); best_st=None; t0=time.time()
    for ep in range(epochs):
        # Train
        model.train(); tot=0; nb=0
        for batch in tl:
            opt.zero_grad()
            if dual:
                v,y,n=[b.float().to(DEVICE) for b in batch]; yl,nl=model(v)
                loss=nn.BCEWithLogitsLoss(pos_weight=pw_t)(yl,y)+nn.BCEWithLogitsLoss()(nl,n)
            else:
                v,i=[b.float().to(DEVICE) for b in batch]
                loss=nn.BCEWithLogitsLoss(pos_weight=pw_t)(model(v),i)
            loss.backward(); opt.step(); tot+=loss.item(); nb+=1
        tl_=tot/max(nb,1)
        # Val
        model.eval(); tot=0; nb=0
        with torch.no_grad():
            for batch in vl:
                if dual:
                    v,y,n=[b.float().to(DEVICE) for b in batch]; yl,nl=model(v)
                    loss=nn.BCEWithLogitsLoss(pos_weight=pw_t)(yl,y)+nn.BCEWithLogitsLoss()(nl,n)
                else:
                    v,i=[b.float().to(DEVICE) for b in batch]
                    loss=nn.BCEWithLogitsLoss(pos_weight=pw_t)(model(v),i)
                tot+=loss.item(); nb+=1
        vl_=tot/max(nb,1); sched.step(); th.append(tl_); vh.append(vl_)
        if vl_<best_val:
            best_val=vl_
            best_st={k:v.cpu().clone() for k,v in model.state_dict().items()}
        eta=(time.time()-t0)/(ep+1)*(epochs-ep-1)
        print(f'  ep {ep+1:3d}/{epochs}  train={tl_:.4f}  val={vl_:.4f}  ETA={eta/60:.1f}хв')
    model.load_state_dict(best_st)
    print(f'  ✓ Best val={best_val:.4f}')
    return th, vh

@torch.no_grad()
def full_inference(model, vol, ps, dual, step=None):
    D,H,W=vol.shape; hp=ps//2; step=step or ps//2
    ym=np.zeros((H,W),np.float32); nm=np.zeros((H,W),np.float32); cnt=np.zeros((H,W),np.float32)
    model.eval()
    tiles=[(py,px) for py in range(hp,H-hp,step) for px in range(hp,W-hp,step)]
    bs=64
    for i in range(0,len(tiles),bs):
        bt=tiles[i:i+bs]; vols=[]; valid=[]
        for py,px in bt:
            p=vol[:,py-hp:py+hp,px-hp:px+hp]
            if p.shape==(D,ps,ps):
                vols.append(torch.from_numpy(p).float().unsqueeze(0)); valid.append((py,px))
        if not vols: continue
        bt2=torch.stack(vols).to(DEVICE)
        if dual:
            yl,nl=model(bt2); yp=torch.sigmoid(yl).cpu().numpy(); np_=torch.sigmoid(nl).cpu().numpy()
        else:
            yp=torch.sigmoid(model(bt2)).cpu().numpy(); np_=np.zeros_like(yp)
        for j,(py,px) in enumerate(valid):
            ym[py-hp:py+hp,px-hp:px+hp]+=yp[j,0]; nm[py-hp:py+hp,px-hp:px+hp]+=np_[j,0]
            cnt[py-hp:py+hp,px-hp:px+hp]+=1
        if (i//bs)%20==0:
            print(f'  {min(i+bs,len(tiles))}/{len(tiles)} ({100*min(i+bs,len(tiles))/len(tiles):.0f}%)')
    cnt=np.maximum(cnt,1)
    if dual: return ((ym/cnt>0.5).astype(int)*2+(nm/cnt>0.5).astype(int)).astype(np.int32)
    return (ym/cnt>0.5).astype(np.int32)

# Перевірка
test=SUBITNet(32,True).to(DEVICE)
x=torch.zeros(1,1,volume.shape[0],64,64).to(DEVICE)
yl,nl=test(x)
print(f'✓ SUBITNet: yang={tuple(yl.shape)}, params={sum(p.numel() for p in test.parameters()):,}')
del test,x; torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════
# КЛІТИНКА 4 — Датасет
# ════════════════════════════════════════════════════════════════
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

PATCH_SIZE      = 64
N_PATCHES       = 4000
DILATION_RADIUS = 4
BATCH_SIZE      = 32

class VesuviusDS(Dataset):
    def __init__(self, vol, ink, mode='subit', ps=64, n=2000, dil=4):
        self.vol=vol; self.ink=ink.astype(np.float32); self.mode=mode
        self.ps=ps; self.hp=ps//2
        if mode=='subit': self.yang,self.yin=make_belnap_labels(ink,dil)
        D,H,W=vol.shape; hp=self.hp; rng=np.random.default_rng(42)
        iy,ix=np.where(ink>0); vy,vx=np.where(ink==0)
        n_ink=int(n*0.4); n_void=n-n_ink
        def sc(ya,xa,nn):
            if len(ya)==0: return []
            idx=rng.integers(0,len(ya),nn*3); res=[]
            for i in idx:
                y,x=ya[i],xa[i]
                if hp<=y<H-hp and hp<=x<W-hp: res.append((y,x))
                if len(res)>=nn: break
            return res[:nn]
        self.coords=sc(iy,ix,n_ink)+sc(vy,vx,n_void)
        print(f'  {mode}: {len(self.coords)} патчів (ink={n_ink}, void={n_void})')

    def __len__(self): return len(self.coords)

    def __getitem__(self,idx):
        py,px=self.coords[idx]; hp=self.hp
        patch=torch.from_numpy(self.vol[:,py-hp:py+hp,px-hp:px+hp]).float().unsqueeze(0)
        if self.mode=='subit':
            return (patch,
                    torch.from_numpy(self.yang[py-hp:py+hp,px-hp:px+hp]).unsqueeze(0),
                    torch.from_numpy(self.yin[py-hp:py+hp,px-hp:px+hp]).unsqueeze(0))
        return patch, torch.from_numpy(self.ink[py-hp:py+hp,px-hp:px+hp]).unsqueeze(0)

def make_loaders(ds, bs=32):
    n=len(ds); idx=list(range(n)); np.random.shuffle(idx); split=int(0.8*n)
    tl=DataLoader(ds,bs,sampler=SubsetRandomSampler(idx[:split]),num_workers=4,pin_memory=True)
    vl=DataLoader(ds,bs,sampler=SubsetRandomSampler(idx[split:]),num_workers=4,pin_memory=True)
    return tl,vl

ratio=(1-ink).sum()/(ink.sum()+1e-8)
print(f'Positive weight: {ratio:.1f}')
print('Датасети:')
ds_s=VesuviusDS(volume,ink,'subit', PATCH_SIZE,N_PATCHES,DILATION_RADIUS)
ds_b=VesuviusDS(volume,ink,'binary',PATCH_SIZE,N_PATCHES,DILATION_RADIUS)
tl_s,vl_s=make_loaders(ds_s,BATCH_SIZE)
tl_b,vl_b=make_loaders(ds_b,BATCH_SIZE)
print(f'SUBIT  train/val: {len(tl_s)}/{len(vl_s)} батчів')
print(f'Binary train/val: {len(tl_b)}/{len(vl_b)} батчів')
print('✓ Готово')


# ════════════════════════════════════════════════════════════════
# КЛІТИНКА 5 — Тренування
# ════════════════════════════════════════════════════════════════
EPOCHS=50; BASE_CH=32

print('='*50)
print('SUBIT-INK (Belnap Loss)')
print('='*50)
sm=SUBITNet(BASE_CH,dual=True)
hs_train,hs_val=train_model(sm,tl_s,vl_s,EPOCHS,True,ratio)

print('\n'+'='*50)
print('Binary Baseline (BCE Loss)')
print('='*50)
bm=SUBITNet(BASE_CH,dual=False)
hb_train,hb_val=train_model(bm,tl_b,vl_b,EPOCHS,False,ratio)

torch.save(sm.state_dict(),'/kaggle/working/subit_model.pth')
torch.save(bm.state_dict(),'/kaggle/working/binary_model.pth')
print('✓ Моделі збережено')


# ════════════════════════════════════════════════════════════════
# КЛІТИНКА 6 — Інференс
# ════════════════════════════════════════════════════════════════
print('Інференс SUBIT-INK...')
bp=full_inference(sm,volume,PATCH_SIZE,dual=True,step=PATCH_SIZE//2)
print('\nІнференс Binary...')
binp=full_inference(bm,volume,PATCH_SIZE,dual=False,step=PATCH_SIZE//2)
np.save('/kaggle/working/belnap_prediction.npy',bp)
np.save('/kaggle/working/binary_prediction.npy',binp)
print('✓ Збережено')


# ════════════════════════════════════════════════════════════════
# КЛІТИНКА 7 — Метрики + Візуалізація
# ════════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BELNAP_COLORS={2:(0,200,255),3:(255,165,0),1:(20,20,20),0:(128,0,128)}
BELNAP_LABELS={2:'INK (T)',3:'BOTH (B) — перевір',1:'VOID (F)',0:'UNKNOWN (N)'}

ink_pred=(bp==2).astype(int); both_pred=(bp==3)
border=binary_dilation(ink.astype(bool),iterations=DILATION_RADIUS)&~ink.astype(bool)
total=bp.size

f1s =f1_score(ink.flatten().astype(int),ink_pred.flatten(),zero_division=0)
prec=precision_score(ink.flatten().astype(int),ink_pred.flatten(),zero_division=0)
rec =recall_score(ink.flatten().astype(int),ink_pred.flatten(),zero_division=0)
f1b =f1_score(ink.flatten().astype(int),binp.flatten().astype(int),zero_division=0)
b_in_both=100*(both_pred&border).sum()/(border.sum()+1e-8)

print(f'\n{"="*50}')
print(f'  РЕЗУЛЬТАТИ — {volume.shape[0]} шарів')
print(f'{"="*50}')
print(f'  SUBIT F1:   {f1s:.4f}')
print(f'  Precision:  {prec:.4f}')
print(f'  Recall:     {rec:.4f}')
print(f'  Binary F1:  {f1b:.4f}')
print(f'  Δ F1:       {f1s-f1b:+.4f}')
print(f'  INK  (T): {100*(bp==2).sum()/total:.1f}%')
print(f'  BOTH (B): {100*(bp==3).sum()/total:.1f}%')
print(f'  VOID (F): {100*(bp==1).sum()/total:.1f}%')
print(f'  UNK  (N): {100*(bp==0).sum()/total:.1f}%')
print(f'  Межі у BOTH: {b_in_both:.1f}%')

iy2,ix2=np.where(ink>0); cy2,cx2=int(iy2.mean()),int(ix2.mean())
SZ=512; sy1=max(0,min(cy2-SZ//2,ink.shape[0]-SZ)); sx1=max(0,min(cx2-SZ//2,ink.shape[1]-SZ))
def crop(a): return a[sy1:sy1+SZ,sx1:sx1+SZ]
mid=volume.shape[0]//2
tif_c=crop(volume[mid]); ink_c=crop(ink); bp_c=crop(bp); bin_c=crop(binp)
ink_only=(bp_c==2).astype(float)
rgb=np.zeros((SZ,SZ,3),dtype=np.uint8)
for s,c in BELNAP_COLORS.items(): rgb[bp_c==s]=c

fig,axes=plt.subplots(2,4,figsize=(22,11))
fig.patch.set_facecolor('#080808')
for ax in axes.flat:
    ax.set_facecolor('#0d0d0d'); [sp.set_edgecolor('#2a2a2a') for sp in ax.spines.values()]
def show(ax,img,t,cmap='gray'):
    ax.imshow(img,cmap=cmap,interpolation='nearest')
    ax.set_title(t,color='#ddd',fontsize=9,pad=5,fontweight='bold'); ax.axis('off')

show(axes[0,0],tif_c,f'Скан сувою (Layer {mid})','bone')
show(axes[0,1],ink_c,'Ground Truth (Vesuvius)','gray')
axes[0,2].imshow(rgb,interpolation='nearest')
axes[0,2].set_title('SUBIT-INK: 4 стани Белнапа',color='#ddd',fontsize=9,fontweight='bold'); axes[0,2].axis('off')
patches=[mpatches.Patch(color=np.array(c)/255,label=BELNAP_LABELS[s]) for s,c in BELNAP_COLORS.items()]
axes[0,2].legend(handles=patches,loc='lower right',fontsize=7,facecolor='#1a1a1a',edgecolor='#555',labelcolor='white')
show(axes[0,3],ink_only,'SUBIT: тільки INK (T)','Blues')
show(axes[1,0],bin_c,'Binary Baseline','gray')
diff=bin_c.astype(int)-ink_only.astype(int)
axes[1,1].imshow(diff,cmap='RdBu',vmin=-1,vmax=1,interpolation='nearest')
axes[1,1].set_title('Різниця: Binary − SUBIT',color='#ddd',fontsize=8,pad=5); axes[1,1].axis('off')
ax=axes[1,2]; ax.set_facecolor('#111')
ax.plot(hs_train,color='#00c8ff',lw=2,label='SUBIT train')
ax.plot(hs_val,  color='#00c8ff',lw=2,ls=':',label='SUBIT val')
ax.plot(hb_train,color='#ff6b35',lw=2,ls='--',label='Binary train')
ax.plot(hb_val,  color='#ff6b35',lw=2,ls=':',label='Binary val')
ax.grid(color='#2a2a2a',lw=0.5); ax.set_title('Криві навчання',color='#ddd',fontsize=9,fontweight='bold')
ax.set_xlabel('Epoch',color='#888'); ax.set_ylabel('Loss',color='#888'); ax.tick_params(colors='#666')
ax.legend(fontsize=7,facecolor='#1a1a1a',edgecolor='#444',labelcolor='white')
ax=axes[1,3]; ax.axis('off')
delta=f1s-f1b
txt=(f'  РЕЗУЛЬТАТИ\n  {volume.shape[0]} шарів\n  {"─"*28}\n'
     f'  SUBIT F1:   {f1s:.4f}\n  Precision:  {prec:.4f}\n  Recall:     {rec:.4f}\n\n'
     f'  Binary F1:  {f1b:.4f}\n  Δ F1:       {delta:+.4f}\n\n'
     f'  Розподіл:\n'
     f'   INK  (T): {100*(bp==2).sum()/total:.1f}%\n'
     f'   BOTH (B): {100*(bp==3).sum()/total:.1f}%\n'
     f'   VOID (F): {100*(bp==1).sum()/total:.1f}%\n'
     f'   UNK  (N): {100*(bp==0).sum()/total:.1f}%\n\n'
     f'  Межі у BOTH: {b_in_both:.1f}%\n  ← галюц. фільтр')
ax.text(0.03,0.97,txt,transform=ax.transAxes,color='#e0e0e0',fontsize=9,va='top',fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.8',facecolor='#151515',edgecolor='#00c8ff',lw=1.5))
plt.suptitle('SUBIT-INK × Vesuvius Challenge — Чотиризначна логіка Белнапа',
             color='white',fontsize=13,fontweight='bold',y=0.998)
plt.tight_layout(rect=[0,0,1,0.985])
plt.savefig('/kaggle/working/subit_ink_result.png',dpi=150,bbox_inches='tight',facecolor='#080808')
plt.show()
print('✓ Готово! Файли у вкладці Output →')
