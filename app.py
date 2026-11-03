import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import glob
import os
import zipfile

# 페이지 기본 설정
st.set_page_config(
    page_title="강릉시 교통안전 통합 의사결정 지원 시스템",
    page_icon="🚦",
    layout="wide"
)

# ---------------------------------------------------------
# 유연한 파일 경로 탐색 함수
# ---------------------------------------------------------
def find_file_flexible(pattern, is_excel=False):
    ext = "*.xlsx" if is_excel else "*.csv"
    search_patterns = [
        f"data/{pattern}{ext}",
        f"data/*{pattern}*{ext}",
        f"{pattern}{ext}",
        f"*{pattern}*{ext}"
    ]
    for p in search_patterns:
        matches = glob.glob(p)
        if matches:
            return matches[0]
    return None

# 1. AI 모델 및 인코더 로드
@st.cache_resource
def load_model_and_encoders():
    model_path = find_file_flexible('accident_risk_model', is_excel=False) or 'models/accident_risk_model.pkl'
    encoder_path = find_file_flexible('label_encoders', is_excel=False) or 'models/label_encoders.pkl'
    
    if os.path.exists('models/accident_risk_model.pkl'):
        model_path = 'models/accident_risk_model.pkl'
    if os.path.exists('models/label_encoders.pkl'):
        encoder_path = 'models/label_encoders.pkl'

    model = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
    return model, encoders

# 2. 강릉시 원본 사고 데이터 로드
@st.cache_data
def load_raw_data():
    raw_path = find_file_flexible('gangneung_data', is_excel=True) or find_file_flexible('강릉시 사고분석', is_excel=True)
    if not raw_path:
        st.error("강릉시 원본 사고 데이터 엑셀 파일을 찾을 수 없습니다.")
        st.stop()
    df = pd.read_excel(raw_path)
    df['lat'] = pd.to_numeric(df['경도'], errors='coerce')
    df['lon'] = pd.to_numeric(df['위도'], errors='coerce')
    df = df.dropna(subset=['lat', 'lon']).copy()
    df = df[(df['lat'] >= 37.5) & (df['lat'] <= 38.0) & (df['lon'] >= 128.5) & (df['lon'] <= 129.2)]
    
    if '발생년월' in df.columns:
        df['연도'] = df['발생년월'].astype(str).str[:4] + '년'
    return df

# 4. 기상청 ASOS 데이터 로드
@st.cache_data
def load_weather_asos_data():
    weather_files = glob.glob('data/OBS_ASOS_TIM_*.csv') + glob.glob('OBS_ASOS_TIM_*.csv')
    w_list = []
    for wf in weather_files:
        try:
            wdf = pd.read_csv(wf, encoding='cp949')
            w_list.append(wdf)
        except Exception:
            pass
    return pd.concat(w_list, ignore_index=True) if w_list else pd.DataFrame()

# 5. 19종 GIS 도로시설물 데이터 로드
@st.cache_data
def load_gis_facilities():
    facility_dfs = {}
    facility_keys = [
        '가로등', '고가도로', '과속방지턱', '교량', '교통표지판', '도로표지판', 
        '미끄럼방지시설', '보안등', '보호구역', '신호등', '육교', '자전거도로', 
        '자전거보관소', '정류장', '주차장', '중앙분리대', '충격흡수시설', '터널', '횡단보도'
    ]
    
    for fac in facility_keys:
        matches = glob.glob(f"data/{fac}*.csv") + glob.glob(f"{fac}*.csv")
        if matches:
            for filepath in matches:
                try:
                    df_fac = pd.read_csv(filepath, encoding='cp949')
                    if '위도' in df_fac.columns and '경도' in df_fac.columns:
                        df_fac['lat'] = pd.to_numeric(df_fac['위도'], errors='coerce')
                        df_fac['lon'] = pd.to_numeric(df_fac['경도'], errors='coerce')
                        df_fac = df_fac.dropna(subset=['lat', 'lon'])
                        df_fac = df_fac[(df_fac['lat'] >= 37.5) & (df_fac['lat'] <= 38.0) & (df_fac['lon'] >= 128.5) & (df_fac['lon'] <= 129.2)]
                        facility_dfs[fac] = df_fac
                        break
                except Exception:
                    pass
    
    zip_matches = glob.glob("data/*.zip") + glob.glob("*.zip")
    if zip_matches:
        try:
            with zipfile.ZipFile(zip_matches[0], 'r') as z:
                for fname in z.namelist():
                    if fname.endswith('.csv'):
                        fac_name = fname.replace('.csv', '').split('/')[-1]
                        if fac_name not in facility_dfs:
                            try:
                                with z.open(fname) as f:
                                    df_fac = pd.read_csv(f, encoding='cp949')
                                    if '위도' in df_fac.columns and '경도' in df_fac.columns:
                                        df_fac['lat'] = pd.to_numeric(df_fac['위도'], errors='coerce')
                                        df_fac['lon'] = pd.to_numeric(df_fac['경도'], errors='coerce')
                                        df_fac = df_fac.dropna(subset=['lat', 'lon'])
                                        df_fac = df_fac[(df_fac['lat'] >= 37.5) & (df_fac['lat'] <= 38.0) & (df_fac['lon'] >= 128.5) & (df_fac['lon'] <= 129.2)]
                                        facility_dfs[fac_name] = df_fac
                            except Exception:
                                pass
        except Exception:
            pass
            
    return facility_dfs

# 데이터 로딩 실행
try:
    model, encoders = load_model_and_encoders()
    df_raw = load_raw_data()
    df_weather = load_weather_asos_data()
    facility_dfs = load_gis_facilities()
except Exception as e:
    st.error(f"데이터 또는 AI 모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# ---------------------------------------------------------
# 집계 보조 함수
# ---------------------------------------------------------

SEVERE = ['중상', '사망']
VALID_OUT = ['사망', '중상', '경상', '상해없음', '부상신고']
ALIASES = {"20세 이하": ["12세 이하", "13-20세", "20세 이하"], "PM": ["개인형이동수단(PM)", "PM"]}
BAD_CLASSES = {"기타불명", "nan", "미분류 - 미분류", ""}
FACTORS = [  # (모델 입력명, 원본 열, 표시명)  모델 입력 순서와 동일
    ('기상상태', '기상상태', '기상 상태'),
    ('노면상태', '노면상태', '노면 상태'),
    ('주야', '주야', '주야 구분'),
    ('도로형태', '도로형태', '도로 형태'),
    ('사고유형', '사고유형', '사고 유형'),
    ('피해운전자 차종', '피해운전자 차종', '피해자 차종'),
    ('피해운전자 연령대', '피해운전자 연령대', '피해자 연령대'),
]


def ordered_options(preferred, classes):
    cls = [str(c) for c in classes if str(c) not in BAD_CLASSES]
    out = [o for o in preferred if o in cls] + [c for c in cls if c not in preferred]
    return out or list(preferred)


def severe_rate(df_all, col, selected):
    valid = df_all[df_all['피해운전자 상해정도'].isin(VALID_OUT)]
    base = valid['피해운전자 상해정도'].isin(SEVERE).mean() * 100
    sub = valid[valid[col].isin(ALIASES.get(selected, [selected]))]
    n = len(sub)
    rate = sub['피해운전자 상해정도'].isin(SEVERE).mean() * 100 if n else np.nan
    return rate, n, base


def facility_coverage(acc, fac_dfs, items, radius=300):
    """사고 지점별로 반경 내 시설이 하나라도 있는 비율(%)과 시설 개수를 반환"""
    from scipy.spatial import cKDTree
    if acc is None or len(acc) == 0:
        return {}
    lat0, lon0 = acc['lat'].mean(), acc['lon'].mean()
    axy = np.column_stack([(acc['lon'] - lon0) * 88000, (acc['lat'] - lat0) * 111000])
    res = {}
    for it in items:
        f = fac_dfs.get(it)
        if f is None or len(f) == 0:
            res[it] = (0.0, 0)
            continue
        fxy = np.column_stack([(f['lon'] - lon0) * 88000, (f['lat'] - lat0) * 111000])
        d, _ = cKDTree(fxy).query(axy)
        res[it] = (float((d <= radius).mean() * 100), int(len(f)))
    return res


def special_accidents(df):
    d = df.copy()
    d['연도'] = d['발생년월'].astype(str).str[:4] + '년'
    rules = {
        '노인 가해운전자(65세 이상)': d['가해운전자 연령대'].astype(str).str.contains('65세', na=False),
        '고령 피해자(65세 이상)': d['피해운전자 연령대'].astype(str).str.contains('65세', na=False),
        '보행자 피해': d['피해운전자 차종'].eq('보행자'),
        '이륜 피해': d['피해운전자 차종'].eq('이륜'),
        '자전거 피해': d['피해운전자 차종'].eq('자전거'),
        '어린이·청소년 피해(20세 이하)': d['피해운전자 연령대'].isin(['12세 이하', '13-20세', '20세 이하']),
    }
    rows = []
    for yr in sorted(d['연도'].dropna().unique()):
        dy = d['연도'] == yr
        for name, m in rules.items():
            s = d[dy & m]
            rows.append({'연도': yr, '유형': name, '사고(건)': len(s), '사망(명)': int(s['사망자수'].sum()),
                         '부상(명)': int(s['중상자수'].sum() + s['경상자수'].sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 격자 계산 (좌표 원점을 고정해 조건이 바뀌어도 같은 격자를 사용)
# ---------------------------------------------------------
LAT0, LON0 = 37.75, 128.90


def cell_ids(lat, lon, cell_m):
    ix = np.floor((np.asarray(lon, dtype=float) - LON0) * 88000 / cell_m).astype(int)
    iy = np.floor((np.asarray(lat, dtype=float) - LAT0) * 111000 / cell_m).astype(int)
    return ix, iy


def cell_center(ix, iy, cell_m):
    return LAT0 + (np.asarray(iy) + 0.5) * cell_m / 111000, LON0 + (np.asarray(ix) + 0.5) * cell_m / 88000


def cell_geojson(cells, cell_m):
    feats = []
    for ix, iy in cells:
        corners = [(ix, iy), (ix + 1, iy), (ix + 1, iy + 1), (ix, iy + 1), (ix, iy)]
        ring = [[LON0 + a * cell_m / 88000, LAT0 + b * cell_m / 111000] for a, b in corners]
        feats.append({"type": "Feature", "id": f"{ix}_{iy}", "properties": {},
                      "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return {"type": "FeatureCollection", "features": feats}


def cell_counts(df_xy, cell_m):
    ix, iy = cell_ids(df_xy['lat'], df_xy['lon'], cell_m)
    return pd.DataFrame({'ix': ix, 'iy': iy}).groupby(['ix', 'iy']).size().rename('c').reset_index()


def abs_reference(df_xy, cell_m):
    """기상×주야 모든 조합 중 한 격자에 쌓인 최대 사고 건수 (조건 간 같은 색 범위를 쓰기 위한 기준)"""
    ix, iy = cell_ids(df_xy['lat'], df_xy['lon'], cell_m)
    t = pd.DataFrame({'w': df_xy['기상상태'].values, 'd': df_xy['주야'].values, 'ix': ix, 'iy': iy})
    return int(t.groupby(['w', 'd', 'ix', 'iy']).size().max())


def risk_grid(df_all, cell_m, a=3.0):
    """격자별 사고 이력과 지점 보정배수(RR).
    같은 구성(차종·연령·도로형태 등)의 사고라면 기대되는 중상 이상 건수(E)와 실제(O)를 비교해
    구성 차이를 제거한 뒤 지점 효과만 남기고, 소표본은 a건만큼 평균(1.0)쪽으로 당겨 보정한다."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    cols = [f[1] for f in FACTORS]
    d = df_all[df_all['피해운전자 상해정도'].isin(VALID_OUT) & df_all[cols].notna().all(axis=1)].copy()
    y = d['피해운전자 상해정도'].isin(SEVERE).astype(int)
    enc = OneHotEncoder(handle_unknown='ignore').fit(d[cols].astype(str))
    lr = LogisticRegression(C=0.5, max_iter=1000).fit(enc.transform(d[cols].astype(str)), y)
    d['E'] = lr.predict_proba(enc.transform(d[cols].astype(str)))[:, 1]
    d['O'] = y
    d['lat'] = pd.to_numeric(d['경도'], errors='coerce')
    d['lon'] = pd.to_numeric(d['위도'], errors='coerce')
    d = d.dropna(subset=['lat', 'lon'])
    d = d[d['lat'].between(37.5, 38.0) & d['lon'].between(128.5, 129.2)]
    d['ix'], d['iy'] = cell_ids(d['lat'], d['lon'], cell_m)
    g = d.groupby(['ix', 'iy']).agg(n=('O', 'size'), O=('O', 'sum'), E=('E', 'sum'),
                                    road=('도로형태', lambda s: s.mode().iat[0]),
                                    acc=('사고유형', lambda s: s.mode().iat[0])).reset_index()
    g['rr'] = ((g['O'] + a) / (g['E'] + a)).clip(0.2, 5)
    return g


def adjust_risk(p_pct, rr):
    """선택 조건의 AI 위험도(%)에 지점 보정배수를 오즈에 곱해 반영"""
    p = min(max(p_pct / 100.0, 0.001), 0.999)
    odds = p / (1 - p) * np.asarray(rr, dtype=float)
    return odds / (1 + odds) * 100


# ---------------------------------------------------------
# 사고 요인 연계 관계도 (연구 단계 분석 방법과 동일: 표준화잔차, Lift, 건수 기준)
# ---------------------------------------------------------
NET_VARS = [('가해차종', '가해운전자 차종'), ('가해성별', '가해운전자 성별'), ('가해연령', '가해운전자 연령대'),
            ('피해차종', '피해운전자 차종'), ('피해성별', '피해운전자 성별'), ('피해연령', '피해운전자 연령대')]
NET_COLORS = {'가해차종': '#C4432E', '가해성별': '#E8892B', '가해연령': '#D9B21F',
              '피해차종': '#0F5C6E', '피해성별': '#2F8F6B', '피해연령': '#7C5CFA'}
_CAR = {'개인형이동수단(PM)': 'PM', '건설기계': '기타', '특수': '기타', '농기계': '기타', '사륜오토바이(ATV)': '기타', '기타불명': '기타'}
_AGE = {'12세 이하': '20세 이하', '13-20세': '20세 이하', '기타불명': np.nan}


def net_value(var, v):
    """사이드바 선택 값을 관계도 범주명으로 변환"""
    if var.endswith('차종'):
        return _CAR.get(v, v)
    if var.endswith('연령'):
        return _AGE.get(v, v)
    return v


def net_prepare(df):
    d = df[df['피해운전자 차종'].notna()].copy()          # 단독사고 제외
    out = pd.DataFrame(index=d.index)
    for name, col in NET_VARS:
        s = d[col]
        if name.endswith('차종'):
            s = s.map(lambda x: _CAR.get(x, x))
        elif name.endswith('연령'):
            s = s.map(lambda x: _AGE.get(x, x))
        else:
            s = s.replace('기타불명', np.nan)
        out[name] = s
    out['중상'] = d['피해운전자 상해정도'].map(lambda x: np.nan if x not in VALID_OUT else float(x in SEVERE))
    return out


def build_network(df, sr_min=2.5, lift_min=1.3, n_min=8):
    d = net_prepare(df)
    names = [n for n, _ in NET_VARS]
    edges = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ct = pd.crosstab(d[a], d[b])
            tot = ct.values.sum()
            exp = np.outer(ct.sum(axis=1), ct.sum(axis=0)) / tot
            sr = (ct.values - exp) / np.sqrt(exp)
            lift = ct.values / exp
            for r, ra in enumerate(ct.index):
                for c, cb in enumerate(ct.columns):
                    if sr[r, c] >= sr_min and lift[r, c] >= lift_min and ct.values[r, c] >= n_min:
                        edges.append({'a': f'{a}:{ra}', 'b': f'{b}:{cb}', 'n': int(ct.values[r, c]),
                                      'lift': float(lift[r, c]), 'sr': float(sr[r, c])})
    edges = pd.DataFrame(edges, columns=['a', 'b', 'n', 'lift', 'sr'])
    rows = []
    for name in names:
        for cat, g in d.groupby(name):
            v = g['중상'].dropna()
            rows.append({'id': f'{name}:{cat}', 'var': name, 'cat': cat, 'n': len(g),
                         'rate': float(v.mean() * 100) if len(v) else np.nan, 'm': len(v)})
    nodes = pd.DataFrame(rows)
    ang = -np.pi / 2 + 2 * np.pi * np.arange(len(nodes)) / len(nodes)
    nodes['x'], nodes['y'], nodes['ang'] = np.cos(ang), np.sin(ang), ang
    return nodes, edges, len(d)


def facility_audit(acc, fac_dfs, items, radius=300):
    """선택한 사고 지점들 기준 시설물 감사: (설치율%, 반경 내 개수(합집합), 전체 개수)"""
    from scipy.spatial import cKDTree
    lat0, lon0 = acc['lat'].mean(), acc['lon'].mean()
    axy = np.column_stack([(acc['lon'] - lon0) * 88000, (acc['lat'] - lat0) * 111000])
    tree_a = cKDTree(axy)
    res = {}
    for it in items:
        f = fac_dfs.get(it)
        if f is None or len(f) == 0:
            res[it] = (0.0, 0, 0)
            continue
        fxy = np.column_stack([(f['lon'] - lon0) * 88000, (f['lat'] - lat0) * 111000])
        d_acc, _ = cKDTree(fxy).query(axy)
        d_fac, _ = tree_a.query(fxy)
        res[it] = (float((d_acc <= radius).mean() * 100), int((d_fac <= radius).sum()), int(len(f)))
    return res


import re

# ---------------------------------------------------------
# 디자인 토큰: 모든 차트와 카드가 같은 색을 쓰도록 한 곳에서 관리
# ---------------------------------------------------------
SEA, SEA_SOFT = "#0F5C6E", "#9CC3CC"
LOW, MID, HIGH = "#2F8F6B", "#D9962B", "#C4432E"
MUTED = "#8A9BA3"
FONT = "Pretendard, 'Noto Sans KR', sans-serif"
MAP_STYLE = "carto-positron"
SEVERITY = {"사망": "#7A1212", "중상": "#E8590C", "경상": "#3B82C4", "부상신고": "#9AA7AE"}
SEV_SIZE = {"사망": 16, "중상": 13, "경상": 10, "부상신고": 8}
FAC_COLORS = ["#12A594", "#7C5CFA", "#D6409F", "#7A8B00", "#8B5E3C"]
TONE_COLOR = {"low": LOW, "mid": MID, "high": HIGH, "na": MUTED}
KEY_ITEMS = ['신호등', '횡단보도', '과속방지턱', '미끄럼방지시설', '보안등', '보호구역']
CUT_MID, CUT_HIGH = 21.9, 40.0   # 강릉시 평균 중상 이상률 / 위험 구간 시작


def html(s):
    """들여쓰기가 마크다운 코드블록으로 해석되지 않도록 줄 앞 공백을 제거"""
    st.markdown("\n".join(l.strip() for l in s.strip().splitlines()), unsafe_allow_html=True)


def md_to_html(s):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)


def style_fig(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=44, b=10),
        font=dict(family=FONT, size=13), title=dict(x=0, font=dict(size=15)),
        legend=dict(orientation="h", y=-0.22, title_text=""),
    )
    return fig


def tone(rate):
    if rate is None or (isinstance(rate, float) and np.isnan(rate)):
        return "na"
    return "high" if rate >= CUT_HIGH else "mid" if rate >= CUT_MID else "low"


# ---------------------------------------------------------
# 전체 사고 데이터(좌표 없는 행 포함): 통계 집계는 반드시 이 데이터를 사용
# ---------------------------------------------------------
@st.cache_data
def load_all_data():
    raw_path = find_file_flexible('gangneung_data', is_excel=True) or find_file_flexible('강릉시 사고분석', is_excel=True)
    return pd.read_excel(raw_path)


@st.cache_data
def cached_special(df):
    return special_accidents(df)


@st.cache_data
def build_quad(df):
    """가해→피해 차종 조합의 건수와 중상 이상률을 원본에서 직접 집계 (표본 15건 이상)"""
    grp = {'개인형이동수단(PM)': 'PM', '건설기계': '기타', '특수': '기타', '농기계': '기타', '사륜오토바이(ATV)': '기타', '기타불명': '기타'}
    d = df[df['피해운전자 차종'].notna() & df['피해운전자 상해정도'].isin(VALID_OUT)].copy()
    d['g'] = d['가해운전자 차종'].map(lambda x: grp.get(x, x))
    d['v'] = d['피해운전자 차종'].map(lambda x: grp.get(x, x))
    d['s'] = d['피해운전자 상해정도'].isin(SEVERE)
    t = d.groupby(['g', 'v'])['s'].agg(['size', 'mean']).reset_index()
    t = t[t['size'] >= 15].copy()
    med = float(t['size'].median())
    base = float(d['s'].mean() * 100)

    def quad(r):
        hf, hr = r['size'] >= med, r['mean'] * 100 >= base
        return ("1사분면 (고빈도·고위험)" if hf and hr else "2사분면 (저빈도·고위험)" if hr
                else "4사분면 (고빈도·저위험)" if hf else "3사분면 (저빈도·저위험)")

    t['사분면'] = t.apply(quad, axis=1)
    t['조합'] = t['g'] + '→' + t['v']
    t = t.rename(columns={'size': '건수'})
    t['중상률'] = (t['mean'] * 100).round(1)
    return t[['조합', '건수', '중상률', '사분면']], med, base


df_all = load_all_data()
df_special = cached_special(df_all)
quad_data, quad_median, base_rate = build_quad(df_all)


@st.cache_data
def cached_network():
    return build_network(df_all)

n_valid = int(df_all['피해운전자 상해정도'].isin(VALID_OUT).sum())


@st.cache_data(show_spinner=False)
def cached_coverage(weather_, day_night_):
    acc = df_raw[(df_raw['기상상태'] == weather_) & (df_raw['주야'] == day_night_)]
    return facility_coverage(acc, facility_dfs, KEY_ITEMS)


# ---------------------------------------------------------
# 스타일
# ---------------------------------------------------------
html("""
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');
html, body, .stApp, .stMarkdown, button, input, label, p, h1, h2, h3, h4 { font-family: 'Pretendard', 'Noto Sans KR', sans-serif; }
.block-container { padding-top: 3.2rem; }
h1 { font-size: 1.85rem !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h2, h3 { font-weight: 650 !important; letter-spacing: -0.01em; }
.stTabs [data-baseweb="tab"] p { font-size: 0.95rem; font-weight: 600; }
[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.28); border-radius: 8px; padding: .7rem 1rem; }
[data-testid="stMetricLabel"] p { font-size: .85rem; }
section[data-testid="stSidebar"] h2 { font-size: 1rem; font-weight: 650; }
.conds { display: flex; flex-wrap: wrap; gap: .4rem; margin: .2rem 0 1.1rem; }
.cond { display: inline-flex; align-items: center; gap: .4rem; padding: .3rem .75rem; border: 1px solid rgba(128,128,128,.3); border-radius: 999px; font-size: .82rem; }
.cond small { opacity: .65; }
.grade { border: 2px solid var(--c); border-radius: 8px; padding: 1.3rem 1.3rem 1rem; min-height: 15rem; background: color-mix(in srgb, var(--c) 8%, transparent); }
.grade .lab { font-size: .9rem; opacity: .75; }
.grade .val { display: flex; align-items: center; gap: .6rem; font-size: 2.7rem; font-weight: 800; color: var(--c); line-height: 1.15; margin: .25rem 0; }
.grade .val small { font-size: 1.1rem; font-weight: 600; opacity: .85; }
.grade .dot { width: .95rem; height: .95rem; border-radius: 50%; background: var(--c); display: inline-block; }
.grade .desc { font-size: .92rem; opacity: .85; }
.seg { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; margin: .9rem 0 .5rem; }
.seg span { padding: .45rem 0; text-align: center; border-radius: 6px; font-size: .85rem; font-weight: 600; background: rgba(128,128,128,.14); opacity: .55; }
.seg span.on { opacity: 1; color: #fff; background: var(--c); }
.dx { border-left: 4px solid #0F5C6E; border-radius: 0 8px 8px 0; background: var(--secondary-background-color); padding: .5rem 1.2rem; min-height: 15rem; }
.dx div { display: grid; grid-template-columns: 3rem 1fr; gap: .6rem; padding: .5rem 0; line-height: 1.6; }
.dx div + div { border-top: 1px solid rgba(128,128,128,.22); }
.dx .k { font-weight: 700; opacity: .7; }
.ft { width: 100%; border-collapse: collapse; font-size: .88rem; }
.ft th { text-align: left; font-weight: 600; opacity: .7; padding: .4rem .5rem; border-bottom: 1px solid rgba(128,128,128,.35); }
.ft td { padding: .5rem; border-bottom: 1px solid rgba(128,128,128,.18); vertical-align: middle; }
.ft small { opacity: .65; margin-left: .35rem; }
.pill { padding: .1rem .6rem; border-radius: 999px; font-weight: 600; }
.pill.low { background: rgba(47,143,107,.16); color: #2F8F6B; }
.pill.mid { background: rgba(217,150,43,.18); color: #C98A1F; }
.pill.high { background: rgba(196,67,46,.16); color: #C4432E; }
.pill.na { background: rgba(128,128,128,.15); }
.ib { display: inline-block; width: 80px; height: 6px; border-radius: 3px; background: rgba(128,128,128,.2); vertical-align: middle; }
.ib i { display: block; height: 100%; border-radius: 3px; background: #0F5C6E; }
.row { display: flex; justify-content: space-between; align-items: center; padding: .5rem 0; border-bottom: 1px solid rgba(128,128,128,.22); }
.chip { font-style: normal; font-size: .78rem; padding: .1rem .55rem; border-radius: 999px; margin-left: .5rem; }
.chip.ok { background: rgba(47,143,107,.16); color: #2F8F6B; }
.chip.warn { background: rgba(196,67,46,.16); color: #C4432E; }
.chip.mid { background: rgba(217,150,43,.18); color: #C98A1F; }
</style>
""")

st.title("강릉시 교통안전 통합 의사결정 지원 시스템")
st.caption("AI 예측 모델 · GIS 19종 도로시설물 · 기상청 ASOS 관측 · 강릉시 사고 원본(2021~2024)을 결합한 대시보드")


# ---------------------------------------------------------
# 사이드바 입력 (선택지는 모델이 학습한 값에서 가져와 항상 유효하도록 구성)
# ---------------------------------------------------------
def opts(col, preferred):
    return ordered_options(preferred, encoders[col].classes_)


st.sidebar.header("사고 환경 조건")
st.sidebar.caption("지도·시설물 감사 탭에는 기상 상태와 주야 구분만 반영됩니다.")

weather = st.sidebar.selectbox("기상 상태", opts('기상상태', ["맑음", "비", "눈", "흐림", "기타"]), key="sb_weather")
surface = st.sidebar.selectbox("노면 상태", opts('노면상태', ["건조", "젖음/습기", "적설", "서리/결빙", "기타"]), key="sb_surface")
day_night = st.sidebar.selectbox("주야 구분", opts('주야', ["주간", "야간"]), key="sb_day_night")
road_type = st.sidebar.selectbox("도로 형태", opts('도로형태', ["교차로 - 교차로안", "교차로 - 교차로부근", "교차로 - 교차로횡단보도내", "단일로 - 기타", "단일로 - 교량", "단일로 - 터널", "단일로 - 고가도로위", "주차장 - 주차장", "기타 - 기타", "기타"]), key="sb_road_type")
accident_type = st.sidebar.selectbox("사고 유형", opts('사고유형', ["차대차 - 충돌", "차대차 - 추돌", "차대차 - 기타", "차대사람 - 횡단중", "차대사람 - 차도통행중", "차대사람 - 기타", "차량단독 - 기타"]), key="sb_accident_type")
victim_type = st.sidebar.selectbox("피해자 차종", opts('피해운전자 차종', ["보행자", "이륜", "자전거", "원동기", "승용", "승합", "화물", "개인형이동수단(PM)", "PM"]), key="sb_victim_type")
victim_age = st.sidebar.selectbox("피해자 연령대", opts('피해운전자 연령대', ["12세 이하", "13-20세", "20세 이하", "21-30세", "31-40세", "41-50세", "51-60세", "61-64세", "65세 이상"]), key="sb_victim_age")

st.sidebar.markdown("---")
st.sidebar.header("기상 관측 보정 (ASOS)")
visibility_input = st.sidebar.slider("시정거리 (m)", min_value=100, max_value=5000, value=3000, step=100, help="1,000m 이하 시 시정장애 위험 가중치 적용")
temp_input = st.sidebar.slider("기온 (°C)", min_value=-15, max_value=35, value=15, step=1)

input_dict = {
    '기상상태': weather, '노면상태': surface, '주야': day_night, '도로형태': road_type,
    '사고유형': accident_type, '피해운전자 차종': victim_type, '피해운전자 연령대': victim_age
}

unknown = [k for k, v in input_dict.items() if str(v) not in set(map(str, encoders[k].classes_))]
if unknown:
    st.warning("모델이 학습하지 않은 값이 선택되어 예측이 부정확할 수 있습니다: " + ", ".join(unknown))

encoded_input = []
for col, val in input_dict.items():
    le = encoders[col]
    encoded_input.append(le.transform([str(val)])[0] if str(val) in le.classes_ else 0)

base_risk_prob = model.predict_proba(np.array(encoded_input).reshape(1, -1))[0][1] * 100

weather_penalty = 0.0
if visibility_input <= 1000:
    weather_penalty += 5.5
if temp_input <= 0 and surface in ["젖음/습기", "서리/결빙"]:
    weather_penalty += 8.0
risk_prob = min(100.0, base_risk_prob + weather_penalty)

# ---------------------------------------------------------
# 요인별 피해도(원본 집계) + 모델 중요도, 시설물 종합평가
# ---------------------------------------------------------
imp = None
if hasattr(model, 'feature_importances_') and len(model.feature_importances_) == len(FACTORS):
    imp = [float(x) for x in model.feature_importances_]

factor_rows = []
for i, (key, col, label) in enumerate(FACTORS):
    rate, n, _ = severe_rate(df_all, col, input_dict[key])
    factor_rows.append(dict(label=label, value=input_dict[key], rate=rate, n=n, imp=imp[i] if imp else None))

coverage = cached_coverage(weather, day_night) if facility_dfs else {}


def facility_text():
    if not coverage:
        return "시설물 데이터가 없어 안전시설 평가는 생략했습니다."
    low = [f"{k} {v[0]:.0f}%" for k, v in coverage.items() if v[1] > 0 and v[0] < 50]
    none = [k for k, v in coverage.items() if v[1] == 0]
    parts = ["사고 지점 반경 300m 내 설치율이 50% 미만인 시설: " + ", ".join(low) + "." if low
             else "핵심 시설 6종 모두 사고 지점의 절반 이상에서 반경 300m 내 확인됩니다."]
    if none:
        parts.append("데이터가 없는 항목: " + ", ".join(none) + ".")
    slip = coverage.get('미끄럼방지시설', (100.0, 0))
    light = coverage.get('보안등', (100.0, 0))
    if weather in ('비', '눈') and slip[1] > 0 and slip[0] < 50:
        parts.append("악천후 조건에서는 미끄럼방지시설 보강이 우선입니다.")
    if day_night == '야간' and light[1] > 0 and light[0] < 50:
        parts.append("야간 조건에서는 조명시설 보강이 우선입니다.")
    return " ".join(parts)


# ---------------------------------------------------------
# 선택 조건 요약 칩
# ---------------------------------------------------------
W_ICON = {"맑음": "☀️", "흐림": "☁️", "비": "🌧️", "눈": "❄️"}
S_ICON = {"건조": "🛣️", "젖음/습기": "💧", "적설": "❄️", "서리/결빙": "🧊"}
V_ICON = {"보행자": "🚶", "이륜": "🏍️", "자전거": "🚲", "원동기": "🛵", "승용": "🚗", "승합": "🚌", "화물": "🚚"}


def chip(icon, label, value):
    return f'<span class="cond">{icon}<small>{label}</small><b>{value}</b></span>'


age_icon = "🧓" if "65세" in victim_age else "🧒" if any(k in victim_age for k in ("12세", "13-20", "20세 이하")) else "🧑"
road_icon = "🚦" if road_type.startswith("교차로") else "🛣️" if road_type.startswith("단일로") else "🅿️" if road_type.startswith("주차장") else "📍"
acc_icon = "🚶" if accident_type.startswith("차대사람") else "🚗" if accident_type.startswith("차대차") else "⚠️"
v_icon = "🛴" if ("PM" in victim_type or "이동수단" in victim_type) else V_ICON.get(victim_type, "🚘")
html('<div class="conds">' + "".join([
    chip(W_ICON.get(weather, "🌫️"), "기상", weather),
    chip(S_ICON.get(surface, "〰️"), "노면", surface),
    chip("🌞" if day_night == "주간" else "🌙", "시간대", day_night),
    chip(road_icon, "도로", road_type),
    chip(acc_icon, "사고", accident_type),
    chip(v_icon, "피해 차종", victim_type),
    chip(age_icon, "피해 연령", victim_age),
    chip("👁️", "시정", f"{visibility_input:,}m"),
    chip("🌡️", "기온", f"{temp_input}°C"),
]) + '</div>')

tab1, tab2, tab_risk, tab3, tab4, tab5 = st.tabs([
    "위험도 예측·진단", "사고 핫스팟 지도", "구간 위험도 지도", "영역 시설물 감사", "4사분면 분석", "취약 유형별 추이"
])

# --- TAB 1: AI 위험도 예측 ---
with tab1:
    if risk_prob >= CUT_HIGH:
        grade, gcolor, gdesc, level = "위험", HIGH, "중상 이상 사고 위험이 높은 구간입니다.", 2
        analysis_text = "보행자/고령자/이륜차 및 악천후 시정 장애 특성이 반영된 **1사분면 최우선 관리(고위험) 수준**입니다."
        action_text = "스마트 가변형 속도제어(VSL 30km/h 제한), 야간 보행자 투광등, 스마트 바닥신호등 및 안개 경고 시스템 투입이 시급합니다."
    elif risk_prob >= CUT_MID:
        grade, gcolor, gdesc, level = "주의", MID, "강릉시 평균 중상 이상률을 웃도는 구간입니다.", 1
        analysis_text = "강릉시 평균 중증 위험도를 상회하며 기상 및 도로 여건에 따른 결합 위험성이 존재합니다."
        action_text = "감속 유도 표지판 추가 배치 및 악천후 시 도로 순찰 강화를 권장합니다."
    else:
        grade, gcolor, gdesc, level = "안전", LOW, "교통사고 위험도가 비교적 낮은 구간입니다.", 0
        analysis_text = "일반 차대차 사고 유형에 해당하며 치명 위험률은 낮으나 총량 관리가 필요한 구간입니다."
        action_text = "교차로 정체 해소 및 회전교차로 전환 등 일반 교통 흐름 개선 사업 조치가 적합합니다."

    fact_text = f"입력 조건 [{weather} | {surface} | {victim_type} | {victim_age} | 시정 {visibility_input}m] 기준 중상 이상 사고 확률은 **{risk_prob:.1f}%**이며 등급은 **{grade}**입니다."
    fac_text = facility_text()

    m1, m2, m3 = st.columns(3)
    m1.metric("AI 기초 위험률", f"{base_risk_prob:.1f}%")
    m2.metric("기상 보정", f"+{weather_penalty:.1f}%p")
    m3.metric("강릉시 평균(21.9%) 대비", f"{risk_prob - CUT_MID:+.1f}%p")

    col1, col2 = st.columns([1, 1.25])
    with col1:
        st.subheader("현재 조건의 위험 등급")
        segs = "".join(f'<span class="{"on" if i == level else ""}">{n}</span>' for i, n in enumerate(["안전", "주의", "위험"]))
        html(f"""
        <div class="grade" style="--c:{gcolor}">
        <div class="lab">중상 이상 사고 확률</div>
        <div class="val"><span class="dot"></span>{grade}<small>{risk_prob:.1f}%</small></div>
        <div class="seg">{segs}</div>
        <div class="desc">{gdesc}</div>
        </div>
        """)
        st.caption("등급 기준: 강릉시 평균 21.9% 미만 안전 · 40% 미만 주의 · 40% 이상 위험")
        if weather_penalty > 0:
            st.warning(f"기상 정밀 관측 보정 적용: 시정거리({visibility_input:,}m)와 기온({temp_input}°C) 영향으로 +{weather_penalty:.1f}%p 상승")

    with col2:
        st.subheader("정리-분석-해석 종합 진단")
        html(f"""
        <div class="dx">
        <div><span class="k">정리</span><span>{md_to_html(fact_text)}</span></div>
        <div><span class="k">분석</span><span>{md_to_html(analysis_text)}</span></div>
        <div><span class="k">해석</span><span>{md_to_html(action_text)}</span></div>
        <div><span class="k">시설</span><span>{fac_text}</span></div>
        </div>
        """)

    st.subheader("조건별 피해도와 모델 중요도")
    mx = max([r['imp'] for r in factor_rows if r['imp'] is not None], default=0) or 1
    body = ""
    for r in factor_rows:
        t = tone(r['rate'])
        rate_txt = "집계 없음" if t == "na" else f"{r['rate']:.1f}%"
        ratio = "-" if t == "na" else f"×{r['rate'] / base_rate:.1f}"
        small = " · 표본 적음" if r['n'] < 30 else ""
        imp_cell = "-" if r['imp'] is None else f'<span class="ib"><i style="width:{r["imp"] / mx * 100:.0f}%"></i></span><small>{r["imp"]:.0%}</small>'
        body += (f'<tr><td>{r["label"]}</td><td><b>{r["value"]}</b></td>'
                 f'<td><span class="pill {t}">{rate_txt}</span><small>n={r["n"]:,}{small}</small></td>'
                 f'<td>{ratio}</td><td>{imp_cell}</td></tr>')
    html('<table class="ft"><tr><th>요인</th><th>선택 값</th><th>중상 이상률</th><th>평균 대비</th><th>모델 중요도</th></tr>' + body + '</table>')
    st.caption(f"중상 이상률은 강릉시 원본 사고(피해운전자 상해 확인 {n_valid:,}건, 평균 {base_rate:.1f}%) 기준이며, 색은 안전·주의·위험 기준과 같습니다.")

    fac_lines = "\n".join(f" - {k}: 사고 지점의 {v[0]:.0f}%에서 반경 300m 내 확인 (전체 {v[1]:,}개소)" for k, v in coverage.items()) or " - 시설물 데이터 없음"
    fac_rows = "\n".join(
        f" - {r['label']} [{r['value']}]: " + ("집계 없음" if tone(r['rate']) == "na" else f"중상 이상률 {r['rate']:.1f}% (평균의 {r['rate'] / base_rate:.1f}배, n={r['n']:,})")
        for r in factor_rows)
    plain = lambda s: s.replace("**", "")
    report_text = f"""[강릉시 교통사고 위험도 AI 진단 리포트]
--------------------------------------------------
■ 진단 대상 조건:
 - 기상 상태: {weather} (시정거리: {visibility_input}m, 기온: {temp_input}°C)
 - 노면 상태: {surface}
 - 주야 구분: {day_night}
 - 도로 형태: {road_type}
 - 사고 유형: {accident_type}
 - 피해자 차종: {victim_type}
 - 피해자 연령대: {victim_age}

■ AI 예측 결과:
 - 최종 중상 이상 사고 확률: {risk_prob:.1f}% → {grade} (기초 {base_risk_prob:.1f}%, 기상보정 +{weather_penalty:.1f}%p)
 - 등급 기준: 평균 21.9% 미만 안전 / 40% 미만 주의 / 40% 이상 위험

■ 요인별 피해도 (강릉시 원본 2021~2024, 평균 {base_rate:.1f}%):
{fac_rows}

■ 안전시설 종합평가 ({weather}·{day_night} 사고 지점 기준, 반경 300m 내 설치율):
{fac_lines}
 - 종합: {fac_text}
 - 참고: 시설의 존재 여부만 확인하며 설치 수준의 적정성은 현장 점검이 필요합니다.

■ 3단계 종합 진단:
 1. 정리: {plain(fact_text)}
 2. 분석: {plain(analysis_text)}
 3. 해석: {plain(action_text)}

--------------------------------------------------
발행처: 강릉시 교통안전 AI 의사결정 지원 시스템
"""
    st.download_button(
        label="진단 리포트 다운로드 (.txt)", data=report_text,
        file_name=f"강릉시_교통사고_진단리포트_{weather}_{victim_type}.txt",
        mime="text/plain", width='stretch'
    )

# --- TAB 2: 사고 핫스팟 지도 ---
NORMS = ["절대 건수 (조건 간 비교)", "조건 내 상대 (최대=100%)", "지점별 조건 비중"]
HEAT_SCALE = [[0, "#FFF1D0"], [0.3, "#F6C667"], [0.65, "#E8742B"], [1, "#A92E22"]]
new_api = hasattr(go, 'Scattermap')
Sc = go.Scattermap if new_api else go.Scattermapbox
Choro = go.Choroplethmap if new_api else go.Choroplethmapbox
lay = 'map' if new_api else 'mapbox'
legend_box = dict(orientation="h", y=0.01, x=0.01, bgcolor="rgba(255,255,255,.88)", font=dict(color="#1B2A33"))


def add_facilities(fig, names):
    for i, name in enumerate(names):
        f = facility_dfs[name]
        fig.add_trace(Sc(lat=f['lat'], lon=f['lon'], mode='markers', hoverinfo='text', text=name,
                         marker=dict(size=6, color=FAC_COLORS[i % len(FAC_COLORS)], opacity=0.85),
                         name=f"{name} ({len(f):,}개소)"))


def grid_figure(cells, values, cell_m, colorscale, zmin, zmax, cbar, customdata, hover, center):
    """격자(사각형) 단위로 값을 칠한 지도를 만든다"""
    geo = cell_geojson(zip(cells['ix'], cells['iy']), cell_m)
    ids = [f"{a}_{b}" for a, b in zip(cells['ix'], cells['iy'])]
    fig = go.Figure(Choro(
        geojson=geo, locations=ids, z=values, colorscale=colorscale, zmin=zmin, zmax=zmax,
        marker=dict(opacity=0.72, line=dict(width=0.4, color="rgba(255,255,255,.7)")),
        colorbar=dict(title=cbar, thickness=12), customdata=customdata, hovertemplate=hover))
    fig.update_layout(**{lay: dict(style=MAP_STYLE, center=dict(lat=center[0], lon=center[1]), zoom=11.5)}, height=520)
    return fig


@st.cache_data(show_spinner=False)
def cached_abs_ref(cell_m):
    return abs_reference(df_raw[['기상상태', '주야', 'lat', 'lon']], cell_m)


@st.cache_data(show_spinner=False)
def cached_risk_grid(cell_m):
    return risk_grid(df_all, cell_m)


with tab2:
    st.subheader("강릉시 사고 핫스팟과 도로시설물 레이어")
    filtered_df = df_raw[(df_raw['기상상태'] == weather) & (df_raw['주야'] == day_night)].copy()
    n_all = int(((df_all['기상상태'] == weather) & (df_all['주야'] == day_night)).sum())

    col_fac1, col_fac2 = st.columns([1, 2])
    with col_fac1:
        st.metric(f"{weather} · {day_night} 지도 표시 사고", f"{len(filtered_df):,}건")
        if n_all:
            st.caption(f"전체 {n_all:,}건 중 좌표가 확인된 {len(filtered_df) / n_all:.0%}입니다.")
    with col_fac2:
        selected_facs = st.multiselect("겹쳐 볼 도로시설물 (여러 개 선택 가능, ✕로 바로 해제)", list(facility_dfs.keys()),
                                       placeholder="시설물을 선택하세요")
        map_mode = st.radio("표시 방식", ["개별 사고점", "격자 히트맵", "부드러운 히트맵"], horizontal=True, key="map_mode")

    cell_m, norm, min_share = 500, NORMS[0], 5
    if map_mode == "격자 히트맵":
        g1, g2, g3 = st.columns(3)
        with g1:
            cell_m = st.select_slider("격자 크기", options=[250, 500, 1000], value=500, format_func=lambda m: f"{m}m", key="hm_cell")
        with g2:
            norm = st.selectbox("색상 기준(표준화)", NORMS, key="hm_norm")
        if norm == NORMS[2]:
            with g3:
                min_share = st.slider("비중 계산 최소 사고 수", 3, 15, 5, key="hm_minn")

    if len(filtered_df) > 0:
        center = (filtered_df['lat'].mean(), filtered_df['lon'].mean())

        if map_mode == "격자 히트맵":
            cond = cell_counts(filtered_df, cell_m)
            allc = cell_counts(df_raw, cell_m).rename(columns={'c': 'all'})
            cond = cond.merge(allc, on=['ix', 'iy'], how='left')
            if norm == NORMS[0]:
                zmax, cbar, cond['z'] = cached_abs_ref(cell_m), "사고 건수", cond['c']
                note = f"모든 조건에서 같은 색 범위(0~{zmax}건)를 씁니다. 조건끼리 비교할 때 사용하세요."
            elif norm == NORMS[1]:
                zmax, cbar, cond['z'] = 100, "조건 내 상대(%)", cond['c'] / cond['c'].max() * 100
                note = "현재 조건에서 가장 많은 격자를 100%로 맞춥니다. 조건 안에서 핫스팟 위치를 볼 때 사용하세요."
            else:
                cond = cond[cond['all'] >= min_share].copy()
                cond['z'] = cond['c'] / cond['all'] * 100
                zmax, cbar = (max(float(cond['z'].max()), 1.0) if not cond.empty else 100), "지점 내 조건 비중(%)"
                note = f"각 격자의 전체 사고 중 현재 조건({weather}·{day_night})의 비중입니다. 지도 전체 평균은 {len(filtered_df) / len(df_raw):.0%}이며, 사고 {min_share}건 미만 격자는 제외했습니다."
            if cond.empty:
                st.warning("기준을 만족하는 격자가 없습니다. 최소 사고 수를 낮추거나 격자 크기를 키워 보세요.")
                fig_map = None
            else:
                fig_map = grid_figure(cond, cond['z'], cell_m, HEAT_SCALE, 0, zmax, cbar,
                                      cond[['c', 'all']].values,
                                      "%{z:.1f}<br>조건 일치 %{customdata[0]}건 / 지점 전체 %{customdata[1]}건<extra></extra>", center)
                st.caption(note)
        elif map_mode == "부드러운 히트맵":
            filtered_df['사고건수'] = 1
            scale = [[0, "rgba(217,150,43,0)"], [0.35, "#D9962B"], [1, HIGH]]
            if hasattr(px, 'density_map'):
                fig_map = px.density_map(filtered_df, lat="lat", lon="lon", z="사고건수", radius=25, center=dict(lat=center[0], lon=center[1]),
                                         zoom=11.5, map_style=MAP_STYLE, color_continuous_scale=scale, opacity=0.85, height=520)
            else:
                fig_map = px.density_mapbox(filtered_df, lat="lat", lon="lon", z="사고건수", radius=25, center=dict(lat=center[0], lon=center[1]),
                                            zoom=11.5, mapbox_style=MAP_STYLE, color_continuous_scale=scale, opacity=0.85, height=520)
            st.caption("색은 현재 조건 안에서의 상대 밀집도입니다. 조건끼리 비교하려면 '격자 히트맵'의 절대 건수 기준을 사용하세요.")
        else:
            fig_map = go.Figure()
            add_facilities(fig_map, selected_facs)   # 시설물을 먼저 그려 사고점이 위에 보이도록 함
            fig_map.add_trace(Sc(
                lat=filtered_df['lat'], lon=filtered_df['lon'], mode='markers', hoverinfo='skip', showlegend=False,
                marker=dict(size=[SEV_SIZE.get(x, 9) + 5 for x in filtered_df['사고내용']], color="white", opacity=0.95)))
            for sev in ["사망", "중상", "경상", "부상신고"]:
                sub = filtered_df[filtered_df['사고내용'] == sev]
                if sub.empty:
                    continue
                fig_map.add_trace(Sc(
                    lat=sub['lat'], lon=sub['lon'], mode='markers', name=f"사고 · {sev} ({len(sub):,}건)",
                    marker=dict(size=SEV_SIZE[sev], color=SEVERITY[sev], opacity=0.95),
                    text=sub['사고유형'], customdata=sub[['법규위반', '피해운전자 차종']].astype(str).values,
                    hovertemplate="%{text}<br>법규위반: %{customdata[0]}<br>피해 차종: %{customdata[1]}<extra>" + sev + "</extra>"))
            fig_map.update_layout(**{lay: dict(style=MAP_STYLE, center=dict(lat=center[0], lon=center[1]), zoom=11.5)}, height=520)
            st.caption("사고점은 흰 테두리가 있는 큰 점(크기·색은 심각도), 시설물은 테두리 없는 작은 점입니다.")

        if fig_map is not None:
            if map_mode != "개별 사고점":
                add_facilities(fig_map, selected_facs)
            fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, font=dict(family=FONT), legend=legend_box)
            st.plotly_chart(fig_map, width='stretch')
    else:
        st.warning("선택한 조건에 해당하는 사고 지점이 없습니다. 기상 상태나 주야 구분을 바꿔 보세요.")

# --- TAB: 구간 위험도 지도 ---
with tab_risk:
    st.subheader("구간 위험도 지도")
    st.caption("선택한 조건의 AI 위험도에, 격자별 과거 사고 경향(같은 구성의 사고와 비교한 보정배수, 소표본은 평균 쪽으로 보정)을 반영한 추정치입니다. 교통량·속도 정보가 없어 도로 한 줄 단위의 실시간 위험은 아닙니다.")
    rk1, rk2, rk3 = st.columns(3)
    with rk1:
        layer = st.radio("지도 층", ["조건 반영 위험도", "중상 이상 사고 이력"], key="rk_layer")
    with rk2:
        cell_r = st.select_slider("격자 크기", options=[250, 500, 1000], value=500, format_func=lambda m: f"{m}m", key="rk_cell")
    with rk3:
        min_n = st.slider("최소 사고 수 (미만 격자는 제외)", 3, 15, 5, key="rk_minn")

    if layer == "조건 반영 위험도":
        grid = cached_risk_grid(cell_r)
        elig = grid[grid['n'] >= min_n].copy()
        if elig.empty:
            st.info("기준을 만족하는 격자가 없습니다. 최소 사고 수를 낮추거나 격자 크기를 키워 보세요.")
        else:
            elig['risk'] = adjust_risk(risk_prob, elig['rr'])
            lo, hi = float(elig['risk'].min()), float(elig['risk'].max())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("선택 조건 위험도", f"{risk_prob:.1f}% · {grade}")
            m2.metric("격자 간 위험도 범위", f"{lo:.1f}~{hi:.1f}%")
            m3.metric("지점 보정배수 범위", f"×{elig['rr'].min():.2f}~{elig['rr'].max():.2f}")
            m4.metric("표시/제외 격자", f"{len(elig)} / {len(grid) - len(elig)}")
            zoom_in = st.checkbox("격자 간 차이 강조 (색 범위를 현재 값 범위에 맞춤)", value=True, key="rk_zoom")
            zmin, zmax = (lo, hi if hi > lo else lo + 1) if zoom_in else (0, 60)
            clat, clon = cell_center(elig['ix'], elig['iy'], cell_r)
            elig['중심 위도'], elig['중심 경도'] = clat, clon
            fig_r = grid_figure(
                elig, elig['risk'], cell_r, [[0, "#E7F2EC"], [0.5, "#F3C87A"], [1, "#C4432E"]], zmin, zmax, "위험도(%)",
                elig[['n', 'O', 'rr']].values,
                "위험도 %{z:.1f}%<br>과거 사고 %{customdata[0]}건 (중상 이상 %{customdata[1]}건)<br>지점 보정 ×%{customdata[2]:.2f}<extra></extra>",
                (float(clat.mean()), float(clon.mean())))
            fig_r.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, font=dict(family=FONT))
            st.plotly_chart(fig_r, width='stretch')
            st.caption("참고 기준: 안전 21.9% 미만 · 주의 40% 미만 · 위험 40% 이상. 격자 간 차이는 조건 자체의 영향보다 작으므로, 색 범위를 강조한 지도는 상대 비교용으로만 보세요.")

            top = elig.sort_values('risk', ascending=False).head(10).copy()
            top['등급'] = np.where(top['risk'] >= CUT_HIGH, '위험', np.where(top['risk'] >= CUT_MID, '주의', '안전'))
            show = pd.DataFrame({
                '순위': range(1, len(top) + 1), '위험도(%)': top['risk'].round(1), '등급': top['등급'],
                '과거 사고(건)': top['n'], '중상 이상(건)': top['O'], '지점 보정': top['rr'].round(2),
                '주요 도로형태': top['road'], '주요 사고유형': top['acc'],
                '중심 위도': top['중심 위도'].round(4), '중심 경도': top['중심 경도'].round(4)})
            st.subheader("위험도 상위 격자")
            st.dataframe(show, width='stretch', hide_index=True)
    else:
        sev_df = filtered_df[filtered_df['피해운전자 상해정도'].isin(SEVERE)]
        st.caption(f"{weather}·{day_night} 조건에서 피해운전자가 중상 이상이었던 사고의 위치입니다(좌표가 있는 사고만).")
        if sev_df.empty:
            st.info("이 조건에는 좌표가 확인된 중상 이상 사고가 없습니다.")
        else:
            sc = cell_counts(sev_df, cell_r)
            sc = sc[sc['c'] >= 1]
            ac = cell_counts(filtered_df, cell_r).rename(columns={'c': 'all'})
            sc = sc.merge(ac, on=['ix', 'iy'], how='left')
            m1, m2 = st.columns(2)
            m1.metric("중상 이상 사고", f"{len(sev_df):,}건")
            st.caption(f"{weather}·{day_night} 지도 표시 사고({len(filtered_df):,}건)의 {len(sev_df) / len(filtered_df):.0%}입니다.")
            m2.metric("해당 격자 수", f"{len(sc)}개")
            fig_s = grid_figure(sc, sc['c'], cell_r, [[0, "#FBE3D6"], [0.5, "#E8742B"], [1, "#7A1212"]], 0, max(int(sc['c'].max()), 1),
                                "중상 이상(건)", sc[['c', 'all']].values,
                                "중상 이상 %{customdata[0]}건 / 조건 사고 %{customdata[1]}건<extra></extra>",
                                (sev_df['lat'].mean(), sev_df['lon'].mean()))
            fig_s.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, font=dict(family=FONT))
            st.plotly_chart(fig_s, width='stretch')


# --- TAB 3: 영역 지정 시설물 감사 ---
def selected_indices(ev):
    """plotly 선택 이벤트에서 선택된 점의 위치를 안전하게 꺼낸다"""
    try:
        sel = ev["selection"] if isinstance(ev, dict) else ev.selection
        pts = sel["point_indices"] if isinstance(sel, dict) else sel.point_indices
        return [int(i) for i in pts]
    except Exception:
        return []


with tab3:
    st.subheader("지도에서 영역을 지정한 시설물 감사")
    st.caption("지도 오른쪽 위 도구 모음에서 사각형(Box) 또는 올가미(Lasso) 선택을 누르고, 감사할 사고 지점을 드래그해 지정하세요. 선택한 사고 지점마다 반경 300m의 도로시설물을 확인합니다.")
    au1, au2 = st.columns([2, 1])
    with au1:
        use_cond = st.checkbox(f"현재 조건({weather}·{day_night})의 사고만 표시", value=True, key="au_cond")
    with au2:
        manual = st.checkbox("좌표로 직접 지정", value=False, key="au_manual", help="지도 선택이 동작하지 않을 때 사용합니다.")

    acc = (df_raw[(df_raw['기상상태'] == weather) & (df_raw['주야'] == day_night)] if use_cond else df_raw).reset_index(drop=True)
    if acc.empty:
        st.warning("표시할 사고 지점이 없습니다. 기상 상태나 주야 구분을 바꿔 보세요.")
    else:
        idx = []
        if manual:
            m1, m2, m3 = st.columns(3)
            with m1:
                mlat = st.number_input("중심 위도", value=float(acc['lat'].mean()), format="%.5f", key="au_lat")
            with m2:
                mlon = st.number_input("중심 경도", value=float(acc['lon'].mean()), format="%.5f", key="au_lon")
            with m3:
                mrad = st.slider("영역 반경 (m)", 100, 2000, 500, step=100, key="au_rad")
            dist = np.sqrt(((acc['lat'] - mlat) * 111000) ** 2 + ((acc['lon'] - mlon) * 88000) ** 2)
            idx = list(np.where(dist <= mrad)[0])
        fig_a = go.Figure(Sc(
            lat=acc['lat'], lon=acc['lon'], mode='markers', showlegend=False,
            marker=dict(size=9, color=[SEVERITY.get(x, MUTED) for x in acc['사고내용']]),
            selected=dict(marker=dict(opacity=1)), unselected=dict(marker=dict(opacity=0.3)),
            text=acc['사고유형'], customdata=acc[['사고내용', '도로형태']].astype(str).values,
            hovertemplate="%{text}<br>%{customdata[0]} · %{customdata[1]}<extra></extra>"))
        fig_a.update_layout(**{lay: dict(style=MAP_STYLE, center=dict(lat=acc['lat'].mean(), lon=acc['lon'].mean()), zoom=11.5)},
                            height=480, margin={"r": 0, "t": 0, "l": 0, "b": 0}, dragmode="select")
        event = st.plotly_chart(fig_a, on_select="rerun", selection_mode=("box", "lasso"), key="audit_map", width='stretch')
        if not manual:
            idx = selected_indices(event)
        idx = [i for i in idx if 0 <= i < len(acc)]

        if not idx:
            st.info("지도에서 사고 지점을 선택하면 감사 결과가 여기에 표시됩니다.")
        elif not facility_dfs:
            st.warning("시설물 데이터가 없어 감사를 수행할 수 없습니다.")
        else:
            sel = acc.iloc[idx]
            audit = facility_audit(sel, facility_dfs, list(facility_dfs.keys()))
            sev_n = int(sel['피해운전자 상해정도'].isin(SEVERE).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("선택한 사고", f"{len(sel):,}건")
            c2.metric("중상 이상 사고", f"{sev_n:,}건")
            c3.metric("주요 도로형태", sel['도로형태'].mode().iat[0])

            col_a, col_b = st.columns([1.2, 1])
            with col_a:
                df_au = pd.DataFrame([{"도로시설물": k, "반경 300m 내 개수": v[1]} for k, v in audit.items()])
                df_au["구분"] = np.where(df_au["도로시설물"].isin(KEY_ITEMS), "핵심 6종", "그 외")
                fig_au = px.bar(df_au, x="반경 300m 내 개수", y="도로시설물", orientation="h", color="구분",
                                color_discrete_map={"핵심 6종": SEA, "그 외": SEA_SOFT}, title="선택 영역 사고 지점 주변 시설물 (개소)")
                fig_au.update_yaxes(categoryorder="total ascending", title_text="")
                st.plotly_chart(style_fig(fig_au, 460), width='stretch')
            with col_b:
                st.subheader("안전시설 공백 진단")
                st.caption("설치율 = 선택한 사고 지점 중 반경 300m 안에 해당 시설이 있는 비율")
                rows = ""
                for item in KEY_ITEMS:
                    pct, cnt, tot = audit.get(item, (0.0, 0, 0))
                    if cnt == 0:
                        chip_html = '<i class="chip warn">보완 필요</i>'
                    elif pct < 50:
                        chip_html = '<i class="chip mid">일부 부족</i>'
                    else:
                        chip_html = '<i class="chip ok">확인</i>'
                    rows += f'<div class="row"><span>{item}</span><span><b>{cnt}</b>개 · 설치율 {pct:.0f}%{chip_html}</span></div>'
                html(f'<div>{rows}</div>')
                slip, light = audit.get('미끄럼방지시설', (100, 1, 1)), audit.get('보안등', (100, 1, 1))
                if weather in ('비', '눈') and slip[0] < 50:
                    st.warning("악천후 조건에서 선택 영역의 미끄럼방지시설이 부족합니다.")
                if day_night == '야간' and light[0] < 50:
                    st.warning("야간 조건에서 선택 영역의 조명시설이 부족합니다.")
                st.caption("시설의 존재 여부만 확인하며, 설치 수준의 적정성은 현장 점검이 필요합니다.")

# --- TAB 4: 4사분면 분석 ---
with tab4:
    st.subheader("가해→피해 차종 조합 4사분면 평가")
    st.caption(f"강릉시 원본 사고에서 직접 집계했습니다(표본 15건 이상 {len(quad_data)}개 조합, 건수 중앙값 {quad_median:.0f}건, 평균 중상 이상률 {base_rate:.1f}%).")

    col_q1, col_q2 = st.columns([1.5, 1])
    with col_q1:
        fig_bubble = px.scatter(
            quad_data, x="건수", y="중상률", size="건수", color="사분면", hover_name="조합",
            hover_data={"건수": True, "중상률": ":.1f", "사분면": True}, log_x=True,
            labels={"건수": "사고 건수 (로그 눈금)", "중상률": "중상 이상률 (%)"},
            color_discrete_map={"1사분면 (고빈도·고위험)": HIGH, "2사분면 (저빈도·고위험)": MID,
                                "3사분면 (저빈도·저위험)": SEA_SOFT, "4사분면 (고빈도·저위험)": SEA})
        fig_bubble.add_hline(y=base_rate, line_dash="dash", line_color=HIGH, annotation_text=f"평균 중상률 {base_rate:.1f}%")
        fig_bubble.add_vline(x=quad_median, line_dash="dash", line_color=MUTED, annotation_text=f"건수 중앙값 {quad_median:.0f}건")
        st.plotly_chart(style_fig(fig_bubble, 440), width='stretch')

    with col_q2:
        fr = pd.DataFrame(factor_rows)
        if imp is None:
            fr['imp'] = [0.04, 0.02, 0.02, 0.08, 0.15, 0.43, 0.26]
            imp_note = "모델에서 중요도를 읽지 못해 기존 고정값을 표시합니다."
        else:
            imp_note = "모델이 학습한 요인별 중요도입니다."
        fr = fr.sort_values('imp')
        fig_imp = go.Figure(go.Bar(
            x=fr['imp'], y=fr['label'], orientation='h',
            marker_color=[TONE_COLOR[tone(r)] for r in fr['rate']],
            text=[f"{v:.0%}" for v in fr['imp']], textposition="outside", cliponaxis=False,
            customdata=[[v, "집계 없음" if tone(r) == "na" else f"{r:.1f}% (n={n:,})"] for v, r, n in zip(fr['value'], fr['rate'], fr['n'])],
            hovertemplate="%{y}<br>중요도 %{x:.1%}<br>선택 값 %{customdata[0]} · 중상 이상률 %{customdata[1]}<extra></extra>"))
        fig_imp.update_layout(title="요인별 모델 중요도와 선택 값의 피해도")
        fig_imp.update_xaxes(visible=False)
        st.plotly_chart(style_fig(fig_imp, 440), width='stretch')
        st.caption(f"막대 길이 = 중요도, 색 = 현재 선택 값의 중상 이상률(초록 21.9% 미만 · 황 40% 미만 · 적 40% 이상). {imp_note}")

    st.divider()
    st.subheader("사고 요인 연계 관계도")
    nodes, edges, n_net = cached_network()
    pos = {r.id: r for r in nodes.itertuples()}
    linked = {'피해차종': net_value('피해차종', victim_type), '피해연령': net_value('피해연령', victim_age)}
    S = {f"{k}:{v}" for k, v in linked.items() if f"{k}:{v}" in pos}

    st.caption(f"왼쪽 사이드바에서 고른 피해자 차종({linked['피해차종']})과 피해자 연령대({linked['피해연령']})가 자동으로 강조됩니다. 아래에서 다른 요인도 골라 연결을 탐색해 보세요.")
    pc = st.columns(4)
    for col, name in zip(pc, ['가해차종', '가해성별', '가해연령', '피해성별']):
        cats = sorted(nodes.loc[nodes['var'] == name, 'cat'])
        with col:
            v = st.selectbox(name, ["(선택 안 함)"] + cats, key=f"net_{name}")
        if v != "(선택 안 함)":
            S.add(f"{name}:{v}")

    def hi(e):
        return (not S) or e.a in S or e.b in S

    fig_net = go.Figure()
    for e in edges.itertuples():
        a, b = pos[e.a], pos[e.b]
        on = hi(e)
        fig_net.add_trace(go.Scatter(
            x=[a.x, (a.x + b.x) / 2, b.x], y=[a.y, (a.y + b.y) / 2, b.y], mode='lines+markers', showlegend=False,
            line=dict(width=(1 + e.sr / 2.2) * (1.0 if on else 0.6), color=HIGH if e.lift >= 2 else MUTED),
            marker=dict(size=10, opacity=0), opacity=0.92 if on else 0.12, hoverinfo='text',
            text=f"{e.a} ↔ {e.b}<br>사고 {e.n}건 · Lift {e.lift:.2f} · 표준화잔차 {e.sr:.1f}"))
    mx = nodes['n'].max()
    for name, g in nodes.groupby('var', sort=False):
        chosen = g['id'].isin(S)
        neighbor = g['id'].map(lambda i: any((e.a == i and e.b in S) or (e.b == i and e.a in S) for e in edges.itertuples()))
        faded = bool(S) & ~(chosen | neighbor)
        fig_net.add_trace(go.Scatter(
            x=g['x'], y=g['y'], mode='markers+text', name=name,
            text=g['cat'], textposition=["middle right" if x > 0.2 else "middle left" if x < -0.2 else ("top center" if y > 0 else "bottom center") for x, y in zip(g['x'], g['y'])],
            textfont=dict(size=11), cliponaxis=False,
            marker=dict(size=[10 + 16 * (n / mx) ** 0.5 + (8 if c else 0) for n, c in zip(g['n'], chosen)],
                        color=NET_COLORS[name], opacity=[0.3 if f else 1 for f in faded],
                        line=dict(width=[3 if c else 0 for c in chosen], color="#111827")),
            customdata=[[n, "집계 없음" if r != r else f"{r:.1f}%"] for n, r in zip(g['n'], g['rate'])],
            hovertemplate="%{text}<br>사고 %{customdata[0]}건 · 피해자 중상 이상률 %{customdata[1]}<extra>" + name + "</extra>"))
    fig_net.update_layout(height=680, margin=dict(l=10, r=10, t=10, b=40), font=dict(family=FONT, size=12),
                          legend=dict(orientation="h", y=-0.04), xaxis=dict(visible=False, range=[-1.6, 1.6]),
                          yaxis=dict(visible=False, range=[-1.35, 1.35], scaleanchor="x"))
    st.plotly_chart(fig_net, width='stretch')
    st.caption(f"연구 단계 분석과 같은 기준(표준화잔차 2.5 이상, Lift 1.3 이상, 사고 8건 이상)으로 원본 {n_net:,}건에서 직접 계산했습니다. 선이 굵을수록 연계가 강하고, 빨간 선은 Lift 2 이상입니다. 연계는 인과관계를 뜻하지 않습니다.")

    shown = edges[edges.apply(hi, axis=1)].sort_values('sr', ascending=False)
    if S:
        st.markdown(f"**선택한 요인과 연결된 조합 {len(shown)}개**")
    if shown.empty:
        st.info("선택한 요인과 기준을 만족하는 연결이 없습니다. 이 요인은 특정 요인과 두드러지게 함께 나타나지 않았다는 뜻입니다.")
    else:
        view = shown.head(12).rename(columns={'a': '요인 A', 'b': '요인 B', 'n': '사고(건)', 'lift': 'Lift', 'sr': '표준화잔차'})
        view['Lift'], view['표준화잔차'] = view['Lift'].round(2), view['표준화잔차'].round(1)
        st.dataframe(view, width='stretch', hide_index=True)


# --- TAB 5: 취약 유형별 사고 추이 ---
with tab5:
    st.subheader("취약 사고 유형별 추이")
    st.caption("원본 사고 2,782건 전체(좌표 유무와 무관)에서 집계했습니다. 유형끼리 겹칠 수 있어 합산이 전체 사고 수와 일치하지 않으며, 원본에 음주운전·뺑소니 항목이 없어 집계하지 않았습니다.")

    if not df_special.empty:
        pal = [SEA, HIGH, MID, "#5B6C8F", MUTED, LOW]
        col_sp1, col_sp2 = st.columns([1.2, 1])
        with col_sp1:
            fig_line = px.line(df_special, x="연도", y="사고(건)", color="유형", markers=True,
                               color_discrete_sequence=pal, title="유형별 연간 사고 건수")
            fig_line.update_xaxes(title_text="")
            st.plotly_chart(style_fig(fig_line, 420), width='stretch')

        with col_sp2:
            tot = df_special.groupby('유형')[['사고(건)', '사망(명)', '부상(명)']].sum().reset_index().sort_values('사고(건)')
            fig_tot = px.bar(tot, x="사고(건)", y="유형", orientation="h", text="사고(건)", hover_data=["사망(명)", "부상(명)"],
                             color_discrete_sequence=[SEA], title="2021~2024 누적 사고 건수 (중복 포함)")
            fig_tot.update_yaxes(title_text="")
            st.plotly_chart(style_fig(fig_tot, 420), width='stretch')

        st.subheader("통합 데이터 요약")
        st.dataframe(df_special, width='stretch', hide_index=True)
    else:
        st.info("집계할 사고 데이터가 없습니다.")