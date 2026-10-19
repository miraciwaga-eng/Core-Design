"""
RC Beam Design (Strength Design Method) — วสท. / ACI 318
หน่วย: kg, cm, ksc (kg/cm²), ton, ton-m

รัน:  streamlit run app.py
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# ค่าคงที่และฐานข้อมูลวัสดุ
# ---------------------------------------------------------------------------
ES = 2.04e6          # ksc  โมดูลัสยืดหยุ่นของเหล็กเสริม
EPS_CU = 0.003       # ความเครียดประลัยของคอนกรีต

MAIN_BARS = {  # ชื่อ: เส้นผ่านศูนย์กลาง (cm)
    "DB10": 1.0, "DB12": 1.2, "DB16": 1.6, "DB20": 2.0,
    "DB25": 2.5, "DB28": 2.8, "DB32": 3.2, "DB36": 3.6,
}
STIRRUP_BARS = {"RB6": 0.6, "RB9": 0.9, "DB10": 1.0, "DB12": 1.2, "DB16": 1.6}
STEEL_GRADES = {"SR24": 2400, "SD30": 3000, "SD40": 4000, "SD50": 5000}
FC_LIST = [180, 210, 240, 280, 300, 320, 350, 400, 450, 500]


def bar_area(db_cm: float) -> float:
    return math.pi * db_cm ** 2 / 4.0


def beta1(fc: float) -> float:
    """ACI 318 / วสท. : fc' ≤ 280 ksc → 0.85, ลดลง 0.05 ทุก 70 ksc, ไม่น้อยกว่า 0.65"""
    if fc <= 280:
        return 0.85
    return max(0.65, 0.85 - 0.05 * (fc - 280) / 70.0)


def phi_flexure(eps_t: float, fy: float) -> float:
    """ACI 318-19 Table 21.2.2 (เหล็กปลอกเดี่ยว)"""
    eps_ty = fy / ES
    if eps_t >= eps_ty + 0.003:
        return 0.90
    if eps_t <= eps_ty:
        return 0.65
    return 0.65 + 0.25 * (eps_t - eps_ty) / 0.003


# ---------------------------------------------------------------------------
# การคำนวณตำแหน่งเหล็กแต่ละชั้น
# ---------------------------------------------------------------------------
def build_layers(layers_in, h, cover, ds, s_v, from_top: bool, fy, tag):
    """คืนค่ารายการชั้นเหล็ก พร้อมระยะ y (cm) วัดจากผิวบน"""
    out = []
    pos = cover + ds  # ระยะจากผิว (ด้านที่เหล็กอยู่) ถึงขอบนอกของชั้นแรก
    prev_db = None
    for i, (bar, n) in enumerate(layers_in):
        if n <= 0:
            continue
        db = MAIN_BARS[bar]
        if prev_db is not None:
            pos += prev_db + s_v
        y_face = pos + db / 2.0
        y = y_face if from_top else h - y_face
        out.append(dict(name=f"{tag}{i + 1}", bar=bar, n=n, db=db,
                        As=n * bar_area(db), y=y, fy=fy))
        prev_db = db
    return out


# ---------------------------------------------------------------------------
# Strain compatibility + สมดุลแรง (หา c โดย bisection)
# ---------------------------------------------------------------------------
def section_forces(c, b, h, fc, b1, layers):
    a = min(b1 * c, h)
    Cc = 0.85 * fc * a * b                    # kg (+ = แรงอัด)
    res = []
    for L in layers:
        eps = EPS_CU * (c - L["y"]) / c       # + = อัด, - = ดึง
        fs = float(np.clip(ES * eps, -L["fy"], L["fy"]))
        # หักพื้นที่คอนกรีตที่ถูกเหล็กแทนที่ (เมื่ออยู่ใน stress block)
        f_net = fs - 0.85 * fc if (L["y"] < a and fs > 0) else fs
        F = L["As"] * f_net
        res.append(dict(L, eps=eps, fs=fs, F=F,
                        yielded=abs(ES * eps) >= L["fy"]))
    total = Cc + sum(r["F"] for r in res)
    return total, a, Cc, res


def solve_section(b, h, fc, layers):
    b1 = beta1(fc)
    lo, hi = 1e-4, 5.0 * h
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f, *_ = section_forces(mid, b, h, fc, b1, layers)
        if f > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-7:
            break
    c = 0.5 * (lo + hi)
    total, a, Cc, res = section_forces(c, b, h, fc, b1, layers)
    # โมเมนต์รอบผิวบน (ΣF = 0 → ได้คู่ควบเดียวกันทุกจุดอ้างอิง)
    Mn = -(Cc * a / 2.0 + sum(r["F"] * r["y"] for r in res))  # kg-cm
    return dict(c=c, a=a, Cc=Cc, layers=res, Mn=abs(Mn), beta1=b1, err=total)


# ---------------------------------------------------------------------------
# ตรวจระยะห่างเหล็ก (วสท./ACI 318 §25.2)
# ---------------------------------------------------------------------------
def spacing_check(layer, b, cover, ds, agg):
    n, db = layer["n"], layer["db"]
    s_min = max(2.5, db, 4.0 / 3.0 * agg)
    clear_w = b - 2 * (cover + ds)
    if n == 1:
        s_act = clear_w - db
    else:
        s_act = (clear_w - n * db) / (n - 1)
    return s_act, s_min, s_act >= s_min - 1e-9


# ---------------------------------------------------------------------------
# วาดรูป
# ---------------------------------------------------------------------------
def draw_section(b, h, cover, ds, all_layers, sol):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2),
                             gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    ax = axes[0]
    ax.add_patch(Rectangle((0, 0), b, h, fc="#e8e8e8", ec="k", lw=1.5))
    # stress block
    a, c = sol["a"], sol["c"]
    ax.add_patch(Rectangle((0, h - a), b, a, fc="#f4a261", alpha=0.45, ec="none"))
    # stirrup
    off = cover + ds / 2
    ax.add_patch(FancyBboxPatch((off, off), b - 2 * off, h - 2 * off,
                                boxstyle=f"round,pad=0,rounding_size={2 * ds}",
                                fc="none", ec="#1d3557", lw=2))
    for L in all_layers:
        n, db = L["n"], L["db"]
        x0 = cover + ds + db / 2
        x1 = b - cover - ds - db / 2
        xs = [b / 2] if n == 1 else np.linspace(x0, x1, n)
        y_plot = h - L["y"]
        color = "#c1121f" if L["y"] > c else "#2a9d8f"
        for x in xs:
            ax.add_patch(Circle((x, y_plot), db / 2, fc=color, ec="k", lw=0.6))
        ax.text(b + 1.2, y_plot, f'{L["name"]}: {n}-{L["bar"]}',
                va="center", fontsize=8)
    if c < h:
        ax.axhline(h - c, color="b", ls="--", lw=1)
        ax.text(-1, h - c, "N.A.", color="b", ha="right", va="center", fontsize=8)
    ax.set_xlim(-8, b + 18)
    ax.set_ylim(-4, h + 4)
    ax.set_aspect("equal")
    ax.set_title(f"Section {b:g} x {h:g} cm")
    ax.axis("off")

    # Strain diagram
    ax = axes[1]
    ys = np.array([0, h])
    eps_line = EPS_CU * (c - ys) / c
    ax.plot(eps_line * 1000, h - ys, "k-")
    ax.axvline(0, color="gray", lw=0.8)
    for r in sol["layers"]:
        ax.plot(r["eps"] * 1000, h - r["y"], "o",
                color="#2a9d8f" if r["eps"] > 0 else "#c1121f")
        ax.annotate(f'{r["name"]}: {r["eps"] * 1000:.2f}', (r["eps"] * 1000, h - r["y"]),
                    textcoords="offset points", xytext=(6, 2), fontsize=8)
    ax.fill_betweenx(h - ys, 0, eps_line * 1000, alpha=0.15)
    ax.set_title("Strain (x10^-3)")
    ax.set_ylim(-4, h + 4)
    ax.set_ylabel("height from bottom (cm)")
    ax.grid(alpha=0.3)

    # Stress in steel
    ax = axes[2]
    names = [r["name"] for r in sol["layers"]]
    fs = [r["fs"] for r in sol["layers"]]
    cols = ["#2a9d8f" if v > 0 else "#c1121f" for v in fs]
    ax.barh(names, fs, color=cols)
    for r in sol["layers"]:
        ax.axvline(r["fy"], color="k", ls=":", lw=0.8)
        ax.axvline(-r["fy"], color="k", ls=":", lw=0.8)
    ax.set_title("Steel stress fs (ksc)  (+comp / -tens)")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RC Beam Design", page_icon="🏗️", layout="wide")
st.title("🏗️ ออกแบบคานคอนกรีตเสริมเหล็ก (วิธีกำลัง)")
st.caption("อ้างอิง มาตรฐาน วสท. (วิธีกำลัง) และ ACI 318-19 · หน่วย kg, cm, ksc, ton-m · "
           "หน่วยแรงในเหล็กแต่ละชั้นคำนวณจาก Strain Compatibility และสมดุลแรง ΣC = ΣT")

with st.sidebar:
    st.header("1) วัสดุ")
    fc = st.selectbox("f'c คอนกรีต (ksc)", FC_LIST, index=FC_LIST.index(240))
    if st.checkbox("กำหนด f'c เอง"):
        fc = st.number_input("f'c (ksc)", 100.0, 800.0, float(fc), 5.0)
    grade = st.selectbox("ชั้นเหล็กยืน", list(STEEL_GRADES), index=2)
    fy = STEEL_GRADES[grade]
    grade_v = st.selectbox("ชั้นเหล็กปลอก", list(STEEL_GRADES), index=0)
    fyt = STEEL_GRADES[grade_v]
    agg = st.number_input("ขนาดมวลรวมโตสุด (cm)", 1.0, 5.0, 2.0, 0.5)

    st.header("2) หน้าตัด")
    b = st.number_input("ความกว้าง b (cm)", 10.0, 200.0, 30.0, 5.0)
    h = st.number_input("ความลึก h (cm)", 15.0, 300.0, 60.0, 5.0)
    cover = st.number_input("ระยะหุ้ม Covering (cm)", 2.0, 10.0, 4.0, 0.5,
                            help="วสท./ACI: คานที่ไม่สัมผัสดิน/อากาศ ≥ 4 cm")
    s_v = st.number_input("ระยะช่องว่างระหว่าง Layer (cm)", 1.0, 10.0, 2.5, 0.5,
                          help="ACI 318 §25.2.2 / วสท.: ≥ 2.5 cm")

    st.header("3) น้ำหนักบรรทุกประลัย")
    Mu = st.number_input("Mu (ton-m)", 0.0, 1000.0, 25.0, 1.0)
    Vu = st.number_input("Vu (ton)", 0.0, 1000.0, 15.0, 1.0)
    L_span = st.number_input("ความยาวช่วงคาน L (m) (ตรวจความลึกขั้นต่ำ)", 0.0, 50.0, 6.0, 0.5)
    support = st.selectbox("สภาพการรองรับ", ["ช่วงเดี่ยว (L/16)", "ต่อเนื่องปลายเดียว (L/18.5)",
                                             "ต่อเนื่องสองปลาย (L/21)", "คานยื่น (L/8)"])

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("เหล็กรับแรงดึง (ล่าง)")
    nb = st.number_input("จำนวน Layer ล่าง", 1, 4, 2, key="nb")
    bot_in = []
    for i in range(int(nb)):
        c1, c2 = st.columns(2)
        bar = c1.selectbox(f"Layer B{i + 1} ขนาด", list(MAIN_BARS), index=4 if i == 0 else 3, key=f"bb{i}")
        n = c2.number_input(f"Layer B{i + 1} จำนวน", 0, 20, 4 if i == 0 else 2, key=f"bn{i}")
        bot_in.append((bar, int(n)))
with col2:
    st.subheader("เหล็กรับแรงอัด (บน)")
    nt = st.number_input("จำนวน Layer บน", 0, 4, 1, key="nt")
    top_in = []
    for i in range(int(nt)):
        c1, c2 = st.columns(2)
        bar = c1.selectbox(f"Layer T{i + 1} ขนาด", list(MAIN_BARS), index=3, key=f"tb{i}")
        n = c2.number_input(f"Layer T{i + 1} จำนวน", 0, 20, 2, key=f"tn{i}")
        top_in.append((bar, int(n)))
with col3:
    st.subheader("เหล็กปลอก")
    st_bar = st.selectbox("ขนาดเหล็กปลอก", list(STIRRUP_BARS), index=1)
    legs = st.number_input("จำนวนขา (legs)", 2, 8, 2)
    s_st = st.number_input("ระยะเรียง s (cm)", 5.0, 60.0, 15.0, 2.5)

ds = STIRRUP_BARS[st_bar]
bot = build_layers(bot_in, h, cover, ds, s_v, from_top=False, fy=fy, tag="B")
top = build_layers(top_in, h, cover, ds, s_v, from_top=True, fy=fy, tag="T")
layers = top + bot

if not bot:
    st.error("ต้องมีเหล็กรับแรงดึงอย่างน้อย 1 Layer")
    st.stop()

sol = solve_section(b, h, fc, layers)
c, a = sol["c"], sol["a"]

# ระยะ d, dt
As_t = sum(r["As"] for r in bot)
d = sum(r["As"] * r["y"] for r in bot) / As_t   # ศูนย์ถ่วงเหล็กล่าง
dt = max(r["y"] for r in bot)                    # ชั้นนอกสุด
eps_t = EPS_CU * (dt - c) / c
phi = phi_flexure(eps_t, fy)
Mn_tm = sol["Mn"] / 1e5
phiMn = phi * Mn_tm

# ---------------------------------------------------------------------------
# ผลลัพธ์: แรงดัด
# ---------------------------------------------------------------------------
st.divider()
st.header("ผลการวิเคราะห์แรงดัด")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("c (cm)", f"{c:.2f}")
m2.metric("a = β₁c (cm)", f"{a:.2f}", help=f"β₁ = {sol['beta1']:.3f}")
m3.metric("εt (ชั้นนอกสุด)", f"{eps_t:.5f}")
m4.metric("φ", f"{phi:.3f}")
m5.metric("φMn (ton-m)", f"{phiMn:.2f}",
          delta=f"{phiMn - Mu:+.2f} เทียบ Mu", delta_color="normal")

df = pd.DataFrame([{
    "Layer": r["name"], "เหล็ก": f'{r["n"]}-{r["bar"]}', "As (cm²)": r["As"],
    "y จากผิวบน (cm)": r["y"], "ε_s": r["eps"], "f_s (ksc)": r["fs"],
    "สถานะ": ("อัด" if r["fs"] > 0 else "ดึง") + (" · ครากแล้ว (fs=fy)" if r["yielded"] else " · ยังไม่คราก"),
    "แรง (ton)": r["F"] / 1000,
} for r in sol["layers"]])
st.dataframe(df.style.format({"As (cm²)": "{:.2f}", "y จากผิวบน (cm)": "{:.2f}",
                              "ε_s": "{:.5f}", "f_s (ksc)": "{:.0f}", "แรง (ton)": "{:.2f}"}),
             width="stretch", hide_index=True)

sumT = -sum(r["F"] for r in sol["layers"] if r["F"] < 0) / 1000
sumC = (sol["Cc"] + sum(r["F"] for r in sol["layers"] if r["F"] > 0)) / 1000
st.markdown(f"**สมดุลแรง:** Cc = {sol['Cc'] / 1000:.2f} t, ΣC = {sumC:.2f} t, ΣT = {sumT:.2f} t "
            f"(ผลต่าง {abs(sumC - sumT):.2e} t) · d = {d:.2f} cm, dt = {dt:.2f} cm · "
            f"Mn = {Mn_tm:.2f} t-m")

st.pyplot(draw_section(b, h, cover, ds, layers, sol))

# ---------------------------------------------------------------------------
# ตรวจสอบตามมาตรฐาน
# ---------------------------------------------------------------------------
checks = []


def chk(item, ok, detail, ref):
    checks.append({"รายการ": item, "ผล": "✅ ผ่าน" if ok else "❌ ไม่ผ่าน",
                   "รายละเอียด": detail, "อ้างอิง": ref})


chk("กำลังรับโมเมนต์", phiMn >= Mu, f"φMn = {phiMn:.2f} ≥ Mu = {Mu:.2f} t-m", "ACI 318-19 §9.5")
As_min = max(0.8 * math.sqrt(fc) / fy, 14.0 / fy) * b * d
chk("เหล็กเสริมน้อยสุด", As_t >= As_min or phiMn >= 4 / 3 * Mu,
    f"As = {As_t:.2f} ≥ As,min = {As_min:.2f} cm² (max(0.8√f'c/fy, 14/fy)·b·d)", "ACI §9.6.1 / วสท.")
chk("ความเครียดเหล็กดึง (คาน)", eps_t >= 0.004,
    f"εt = {eps_t:.5f} ≥ 0.004  ({'Tension-controlled' if eps_t >= fy / ES + 0.003 else 'Transition'})",
    "ACI §9.3.3.1")

for L in layers:
    s_act, s_min, ok = spacing_check(L, b, cover, ds, agg)
    chk(f"ระยะห่างเหล็กในแนวนอน {L['name']}", ok,
        f"ช่องว่างจริง = {s_act:.2f} ≥ {s_min:.2f} cm (max(2.5, db, 4/3·agg))", "ACI §25.2.1")
if (len(bot) > 1) or (len(top) > 1):
    s_v_min = max(2.5, max(L["db"] for L in layers))
    chk("ระยะช่องว่างระหว่าง Layer", s_v >= 2.5,
        f"s = {s_v:.2f} ≥ 2.5 cm (แนะนำ ≥ {s_v_min:.1f} cm)", "ACI §25.2.2")
if top and bot:
    top_edge = max(L["y"] + L["db"] / 2 for L in top)
    bot_edge = min(L["y"] - L["db"] / 2 for L in bot)
    chk("เหล็กบน/ล่างไม่ซ้อนทับกัน", bot_edge - top_edge >= 2.5,
        f"ช่องว่าง = {bot_edge - top_edge:.2f} cm", "-")

if L_span > 0:
    div = {"ช่วงเดี่ยว (L/16)": 16, "ต่อเนื่องปลายเดียว (L/18.5)": 18.5,
           "ต่อเนื่องสองปลาย (L/21)": 21, "คานยื่น (L/8)": 8}[support]
    h_min = L_span * 100 / div * (0.4 + fy / 7000)
    chk("ความลึกขั้นต่ำ (ไม่ตรวจการแอ่นตัว)", h >= h_min,
        f"h = {h:.1f} ≥ {h_min:.1f} cm (L/{div}·(0.4+fy/7000))", "ACI Table 9.3.1.1")

# ---------------------------------------------------------------------------
# แรงเฉือน
# ---------------------------------------------------------------------------
phi_v = 0.75
Av = legs * bar_area(ds)
Vc = 0.53 * math.sqrt(fc) * b * d / 1000                 # ton
Vs = Av * fyt * d / s_st / 1000                          # ton
Vs_max = 2.1 * math.sqrt(fc) * b * d / 1000
phiVn = phi_v * (Vc + min(Vs, Vs_max))
s_max = min(d / 2, 60.0) if Vs <= 1.1 * math.sqrt(fc) * b * d / 1000 else min(d / 4, 30.0)
Av_min = max(0.2 * math.sqrt(fc), 3.5) * b * s_st / fyt
Vs_req = max(Vu / phi_v - Vc, 0.0)
s_req = Av * fyt * d / (Vs_req * 1000) if Vs_req > 0 else float("inf")

st.divider()
st.header("ผลการออกแบบแรงเฉือน")
v1, v2, v3, v4, v5 = st.columns(5)
v1.metric("Av (cm²)", f"{Av:.3f}", help=f"{legs} ขา {st_bar}")
v2.metric("Vc (ton)", f"{Vc:.2f}", help="0.53√f'c·b·d")
v3.metric("Vs (ton)", f"{Vs:.2f}", help="Av·fyt·d / s")
v4.metric("φVn (ton)", f"{phiVn:.2f}", delta=f"{phiVn - Vu:+.2f} เทียบ Vu")
v5.metric("s ที่ต้องการ (cm)", "-" if math.isinf(s_req) else f"{s_req:.1f}")

chk("กำลังรับแรงเฉือน", phiVn >= Vu, f"φVn = {phiVn:.2f} ≥ Vu = {Vu:.2f} t (φ = 0.75)", "ACI §22.5")
chk("Vs สูงสุด (ขนาดหน้าตัด)", Vs_req <= Vs_max,
    f"Vs,req = {Vs_req:.2f} ≤ 2.1√f'c·b·d = {Vs_max:.2f} t", "ACI §22.5.1.2")
chk("ระยะเรียงเหล็กปลอกสูงสุด", s_st <= s_max, f"s = {s_st:.1f} ≤ {s_max:.1f} cm", "ACI §9.7.6.2.2")
if Vu > 0.5 * phi_v * Vc:
    chk("เหล็กปลอกน้อยสุด", Av >= Av_min,
        f"Av = {Av:.3f} ≥ {Av_min:.3f} cm² (max(0.2√f'c, 3.5)·b·s/fyt)", "ACI §9.6.3")

st.divider()
st.header("สรุปการตรวจสอบ")
cdf = pd.DataFrame(checks)
st.dataframe(cdf, width="stretch", hide_index=True)
if all("ผ่าน" in x and "ไม่" not in x for x in cdf["ผล"]):
    st.success("✅ หน้าตัดผ่านการตรวจสอบทุกรายการ")
else:
    st.warning("⚠️ มีรายการที่ไม่ผ่าน — ปรับขนาดหน้าตัด/เหล็กเสริม")

with st.expander("📘 สมมติฐานและสมการที่ใช้"):
    st.markdown(rf"""
- **Whitney stress block:** $C_c = 0.85 f'_c\, a\, b$, $a = \beta_1 c$, $\beta_1 = 0.85$ เมื่อ $f'_c \le 280$ ksc
  ลดลง 0.05 ทุก 70 ksc ไม่น้อยกว่า 0.65 (ปัจจุบัน $\beta_1 = {sol['beta1']:.3f}$)
- **Strain compatibility:** $\varepsilon_{{si}} = 0.003\,(c - y_i)/c$ สำหรับเหล็กแต่ละ Layer
- **หน่วยแรงเหล็กแต่ละ Layer:** $f_{{si}} = E_s\varepsilon_{{si}}$ โดย $|f_{{si}}| \le f_y$, $E_s = 2.04\times10^6$ ksc
  — เหล็กแต่ละชั้นจึงอาจคราก ($f_s = f_y$) หรือไม่คราก ($f_s < f_y$) ขึ้นกับตำแหน่ง
- **สมดุลแรง:** $C_c + \sum A_{{si}} f_{{si}} = 0$ แก้หา $c$ ด้วยวิธี Bisection
  (เหล็กอัดที่อยู่ใน stress block หักพื้นที่คอนกรีต: $f_s - 0.85f'_c$)
- **φ แรงดัด:** 0.90 เมื่อ $\varepsilon_t \ge \varepsilon_{{ty}} + 0.003$, 0.65 เมื่อ $\varepsilon_t \le \varepsilon_{{ty}}$, เส้นตรงระหว่างนั้น
- **แรงเฉือน:** $V_c = 0.53\sqrt{{f'_c}}\,b\,d$, $V_s = A_v f_{{yt}} d/s$, φ = 0.75
- ระยะ d = ศูนย์ถ่วงเหล็กดึงทุก Layer, dt = ระยะถึงเหล็กดึงชั้นนอกสุด
""")
