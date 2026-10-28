# SUBIT-INK 🏺

**Чотиризначна логіка Белнапа для детекції чорнила в Геркуланських сувоях**

> Contribution до [Vesuvius Challenge](https://scrollprize.org) · Progress Prize submission

---

## Проблема

Стандартна детекція чорнила — бінарна: піксель або є чорнилом, або ні.  
Але модель **не може виразити невизначеність**. Це призводить до:
- Галюцинацій на деградованих ділянках
- Відсутності інформації про межі букв
- Неможливості розрізнити "немає даних" від "точно не чорнило"

## Рішення

SUBIT-INK замінює бінарний вихід на **4 стани білаттису Белнапа**:

| Стан | Символ | Значення | Колір |
|------|--------|----------|-------|
| **INK** | T (yang=1, yin=0) | Впевнено чорнило | 🔵 блакитний |
| **BOTH** | B (yang=1, yin=1) | Суперечливий / межа букви | 🟠 помаранчевий |
| **VOID** | F (yang=0, yin=1) | Впевнено не чорнило | ⚫ темний |
| **UNKNOWN** | N (yang=0, yin=0) | Відсутність даних / лакуна | 🟣 фіолетовий |

## Зміна в коді — мінімальна

```python
# До: стандартний бінарний вихід
self.head = nn.Conv2d(features, 1, kernel_size=1)
logit = self.head(features)
loss = BCEWithLogitsLoss()(logit, binary_label)

# Після: SUBIT dual head
self.yang_head = nn.Conv2d(features, 1, kernel_size=1)
self.yin_head  = nn.Conv2d(features, 1, kernel_size=1)
yang_logit, yin_logit = self.yang_head(f), self.yin_head(f)
loss = BelnapLoss()(yang_logit, yin_logit, yang_label, yin_label)
```

**Один додатковий Conv2d шар. Той самий GPU-стек. Нова семантика.**

## Встановлення

```bash
pip install torch numpy scipy scikit-learn matplotlib
git clone https://github.com/your-org/subit-ink
cd subit-ink
```

## Швидкий старт

```python
from subit_core import binary_to_belnap_labels, decode_belnap, BelnapState
from subit_model import SUBITInkDetector

# Перетворення бінарних анотацій у мітки Белнапа
yang_labels, yin_labels = binary_to_belnap_labels(
    ink_mask,           # стандартна маска з inklabels.png
    dilation_radius=3   # ширина зони BOTH навколо букв
)

# Модель
model = SUBITInkDetector(in_depth=65)  # 65 шарів surface volume

# Inference → карта Белнапа
belnap_map = model.predict_belnap(surface_volume_tensor)

# Фільтр галюцинацій: беремо лише впевнені пікселі
confident_ink = (belnap_map == BelnapState.INK)
review_needed = (belnap_map == BelnapState.BOTH)   # для анотаторів
data_gaps     = (belnap_map == BelnapState.UNKNOWN) # лакуни в тексті
```

## Файли

```
subit_core.py   — Базова логіка: білаттис Белнапа, метрики, loss
subit_model.py  — 3D U-Net із SUBIT dual head + binary baseline
demo.py         — Демонстрація на синтетичних даних
```

## Результати на синтетичних даних

| Метрика | Binary Baseline | SUBIT-INK |
|---------|----------------|-----------|
| F1 (впевнений INK) | 0.877 | 0.754 |
| Precision | — | 0.815 |
| Recall | — | 0.702 |
| Межі букв у BOTH | — | **95%** |
| Галюцинаційний фільтр | ❌ | ✅ BOTH-клас |

> Примітка: нижчий F1 в SUBIT на синтетичних даних очікуваний —  
> модель виводить частину пікселів у BOTH (перевір), а не в INK.  
> Це і є ціль: менше хибних впевнених передбачень.

## Теоретична основа

SUBIT-INK базується на **SUBIT-TOPOS** — формальній теорії  
самореферентних динамічних систем з 4-значною алгеброю Белнапа.

Відповідність теорії і фізики сувою:

| Ω_SUBIT | SUBIT-INK | Фізичний сенс |
|---------|-----------|---------------|
| `stable` | INK (T) | Чорнило стабільне у всіх шарах 3D-скану |
| `metastable` | BOTH (B) | Межа букви, нестабільний сигнал |
| `cyclic` | UNKNOWN (N) | Сигнал з'являється і зникає — артефакт |
| `chaotic` | VOID (F) | Відсутній систематичний сигнал |

Центральна теза SUBIT: **P є істинним ⟺ F(P) ⊆ P**  
В контексті детекції: піксель є чорнилом лише якщо  
він стабільно виявляється у суміжних шарах.

## Ліцензія

MIT License — повністю open source, відповідно до вимог Vesuvius Challenge.

---

*SUBIT-INK · 2026 · Vesuvius Challenge Progress Prize Submission*
