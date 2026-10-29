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
    page_icon="🚨",
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

# 3. 강릉시 원본 데이터에서 6대 특수 사고 유형 자동 필터링 및 집계
@st.cache_data
def extract_gangneung_special_accidents(df_raw):
    records = []
    
    if '연도' not in df_raw.columns and '발생년월' in df_raw.columns:
        df_raw['연도'] = df_raw['발생년월'].astype(str).str[:4] + '년'
        
    years = sorted(df_raw['연도'].dropna().unique())
    
    for yr in years:
        df_yr = df_raw[df_raw['연도'] == yr]
        
        # 1. 음주운전
        if '법규위반' in df_yr.columns:
            df_dunk = df_yr[df_yr['법규위반'].str.contains('음주', na=False)]
            records.append({'연도': yr, '사고(건)': len(df_dunk), '사망(명)': df_dunk['사망자수'].sum() if '사망자수' in df_dunk.columns else 0, '부상(명)': (df_dunk['중상자수'].sum() + df_dunk['경상자수'].sum()) if '중상자수' in df_dunk.columns else 0, '유형': '음주운전'})
            
        # 2. 노인운전자 (65세 이상)
        if '가해운전자 연령대' in df_yr.columns:
            df_elder = df_yr[df_yr['가해운전자 연령대'].str.contains('65세', na=False)]
            records.append({'연도': yr, '사고(건)': len(df_elder), '사망(명)': df_elder['사망자수'].sum() if '사망자수' in df_elder.columns else 0, '부상(명)': (df_elder['중상자수'].sum() + df_elder['경상자수'].sum()) if '중상자수' in df_elder.columns else 0, '유형': '노인운전자'})
            
        # 3. 보행자 사고
        if '피해운전자 차종' in df_yr.columns:
            df_ped = df_yr[df_yr['피해운전자 차종'].str.contains('보행자', na=False)]
            records.append({'연도': yr, '사고(건)': len(df_ped), '사망(명)': df_ped['사망자수'].sum() if '사망자수' in df_ped.columns else 0, '부상(명)': (df_ped['중상자수'].sum() + df_ped['경상자수'].sum()) if '중상자수' in df_ped.columns else 0, '유형': '보행자'})

        # 4. 어린이 사고 (20세 이하)
        if '피해운전자 연령대' in df_yr.columns:
            df_child = df_yr[df_yr['피해운전자 연령대'].str.contains('20세 이하', na=False)]
            records.append({'연도': yr, '사고(건)': len(df_child), '사망(명)': df_child['사망자수'].sum() if '사망자수' in df_child.columns else 0, '부상(명)': (df_child['중상자수'].sum() + df_child['경상자수'].sum()) if '중상자수' in df_child.columns else 0, '유형': '어린이(20세이하)'})

        # 5. 뺑소니
        if '사고유형' in df_yr.columns:
            df_hit = df_yr[df_yr['사고유형'].str.contains('뺑소니|도주', na=False)]
            records.append({'연도': yr, '사고(건)': len(df_hit), '사망(명)': df_hit['사망자수'].sum() if '사망자수' in df_hit.columns else 0, '부상(명)': (df_hit['중상자수'].sum() + df_hit['경상자수'].sum()) if '중상자수' in df_hit.columns else 0, '유형': '뺑소니'})

    return pd.DataFrame(records)

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
    df_special = extract_gangneung_special_accidents(df_raw)
    df_weather = load_weather_asos_data()
    facility_dfs = load_gis_facilities()
except Exception as e:
    st.error(f"데이터 또는 AI 모델을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 메인 타이틀
st.title("🚨 강릉시 교통안전 통합 의사결정 지원 시스템")
st.caption("AI 예측 모델 + GIS 19종 도로시설물 + 기상청 ASOS 관측 데이터 + 강릉시 특수 사고 지표 결합 대시보드")
st.markdown("---")

# 사이드바 입력
st.sidebar.header("⚙️ 사고 환경 조건 설정")

weather = st.sidebar.selectbox("기상 상태", options=["맑음", "비", "눈", "흐림", "기타"], key="sb_weather")
surface = st.sidebar.selectbox("노면 상태", options=["건조", "젖음/습기", "적설", "서리/결빙", "기타"], key="sb_surface")
day_night = st.sidebar.selectbox("주야 구분", options=["주간", "야간"], key="sb_day_night")
road_type = st.sidebar.selectbox("도로 형태", options=["교차로 - 교차로안", "교차로 - 교차로부근", "단일로 - 기타", "단일로 - 교량", "주차장 - 주차장", "기타"], key="sb_road_type")
accident_type = st.sidebar.selectbox("사고 유형", options=["차대차 - 충돌", "차대차 - 추돌", "차대사람 - 횡단중", "차대사람 - 차도통행중", "차량단독 - 기타"], key="sb_accident_type")
victim_type = st.sidebar.selectbox("피해자 차종", options=["보행자", "이륜", "자전거", "원동기", "승용", "승합", "화물", "PM"], key="sb_victim_type")
victim_age = st.sidebar.selectbox("피해자 연령대", options=["20세 이하", "21-30세", "31-40세", "41-50세", "51-60세", "61-64세", "65세 이상"], key="sb_victim_age")

st.sidebar.markdown("---")
st.sidebar.header("🌤️ 기상청 정밀 관측 연동 (ASOS)")
visibility_input = st.sidebar.slider("시정거리 (m)", min_value=100, max_value=5000, value=3000, step=100, help="1,000m 이하 시 시정장애 위험 가중치 적용")
temp_input = st.sidebar.slider("기온 (°C)", min_value=-15, max_value=35, value=15, step=1)

# AI 위험도 예측 수행
input_dict = {
    '기상상태': weather,
    '노면상태': surface,
    '주야': day_night,
    '도로형태': road_type,
    '사고유형': accident_type,
    '피해운전자 차종': victim_type,
    '피해운전자 연령대': victim_age
}

encoded_input = []
for col, val in input_dict.items():
    le = encoders[col]
    encoded_val = le.transform([str(val)])[0] if str(val) in le.classes_ else 0
    encoded_input.append(encoded_val)

base_risk_prob = model.predict_proba(np.array(encoded_input).reshape(1, -1))[0][1] * 100

# 기상 보정
weather_penalty = 0.0
if visibility_input <= 1000:
    weather_penalty += 5.5
if temp_input <= 0 and surface in ["젖음/습기", "서리/결빙"]:
    weather_penalty += 8.0

risk_prob = min(100.0, base_risk_prob + weather_penalty)

# 메인 탭 구획 (5개 탭)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🚨 AI 위험도 예측 및 진단", 
    "🗺️ 사고 핫스팟 & 19종 GIS 시설물 지도", 
    "🔍 반경 300m 안전시설물 감사(Audit)",
    "📊 4사분면 & 통계 분석",
    "📈 강릉시 취약 유형별 사고 추이"
])

# --- TAB 1: AI 위험도 예측 ---
with tab1:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("🎯 예측 위험도 (중상 이상 확률)")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_prob,
            number={'suffix': "%", 'font': {'size': 38}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1F4E79"},
                'steps': [
                    {'range': [0, 21.9], 'color': "#E2EFDA"},
                    {'range': [21.9, 40], 'color': "#FFF2CC"},
                    {'range': [40, 100], 'color': "#FCE4D6"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 21.9}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, width='stretch')
        
        if weather_penalty > 0:
            st.warning(f"⚠️ **기상 정밀 관측 보정 적용:** 시정거리({visibility_input}m) 및 기온({temp_input}°C) 영향으로 위험도 +{weather_penalty:.1f}%p 상승")
        else:
            st.info("💡 **강릉시 평균 중상 이상률:** 21.9% (빨간 선 기준)")

    with col2:
        st.subheader("📋 '정리 - 분석 - 해석' 3단계 종합 진단")
        fact_text = f"입력 조건 [{weather} | {surface} | {victim_type} | {victim_age} | 시정 {visibility_input}m] 기준 최종 치명 위험률은 **{risk_prob:.1f}%**입니다."
        
        if risk_prob >= 40:
            analysis_text = "보행자/고령자/이륜차 및 악천후 시정 장애 특성이 반영된 **1사분면 최우선 관리(고위험) 수준**입니다."
            action_text = "스마트 가변형 속도제어(VSL 30km/h 제한), 야간 보행자 투광등, 스마트 바닥신호등 및 안개 경고 시스템 투입이 시급합니다."
        elif risk_prob >= 21.9:
            analysis_text = "강릉시 평균 중증 위험도를 상회하며 기상 및 도로 여건에 따른 결합 위험성이 존재합니다."
            action_text = "감속 유도 표지판 추가 배치 및 악천후 시 도로 순찰 강화를 권장합니다."
        else:
            analysis_text = "일반 차대차 사고 유형에 해당하며 치명 위험률은 낮으나 총량 관리가 필요한 구간입니다."
            action_text = "교차로 정체 해소 및 회전교차로 전환 등 일반 교통 흐름 개선 사업 조치가 적합합니다."

        st.markdown(f"""
        <div style="background-color: #F8F9FA; padding: 18px; border-radius: 8px; border-left: 6px solid #1F4E79;">
            <p style="margin-bottom:10px;"><b>1. 정리 (Fact):</b> {fact_text}</p>
            <p style="margin-bottom:10px;"><b>2. 분석 (Analysis):</b> {analysis_text}</p>
            <p style="margin-bottom:12px;"><b>3. 해석 (So What):</b> {action_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
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
 - 최종 중상 이상 위험률: {risk_prob:.1f}% (기초 위험률: {base_risk_prob:.1f}%, 기상보정: +{weather_penalty:.1f}%p)

■ 3단계 종합 진단:
 1. 정리 (Fact): {fact_text}
 2. 분석 (Analysis): {analysis_text}
 3. 해석 (So What): {action_text}

--------------------------------------------------
발행처: 강릉시 교통안전 AI 의사결정 지원 시스템
"""
        st.download_button(
            label="📥 AI 진단 리포트 다운로드 (.txt)",
            data=report_text,
            file_name=f"강릉시_교통사고_진단리포트_{weather}_{victim_type}.txt",
            mime="text/plain",
            width='stretch'
        )

# --- TAB 2: 지도 시각화 ---
with tab2:
    st.subheader("🗺️ 강릉시 사고 핫스팟 & 19종 GIS 도로시설물 선택 레이어")
    filtered_df = df_raw[(df_raw['기상상태'] == weather) & (df_raw['주야'] == day_night)].copy()
    
    col_fac1, col_fac2 = st.columns([1, 2])
    with col_fac1:
        st.write(f"📍 검색 조건 **[{weather} + {day_night}]** 실제 사고: **{len(filtered_df)}건**")
        selected_fac = st.selectbox("🗺️ 오버레이할 도로시설물 선택:", ["(선택 없음)"] + list(facility_dfs.keys()))
    
    with col_fac2:
        map_style = st.radio("지도 시각화 방식 선택:", ["밀집도 히트맵 (Heatmap)", "개별 사고점 (Scatter)"], horizontal=True, key="map_toggle")
    
    if len(filtered_df) > 0:
        center_lat = filtered_df['lat'].mean()
        center_lon = filtered_df['lon'].mean()

        if map_style == "밀집도 히트맵 (Heatmap)":
            filtered_df['사고건수'] = 1
            if hasattr(px, 'density_map'):
                fig_map = px.density_map(
                    filtered_df, lat="lat", lon="lon", z="사고건수", radius=25,
                    center=dict(lat=center_lat, lon=center_lon), zoom=11.5,
                    map_style="open-street-map", color_continuous_scale="YlOrRd", opacity=0.8, height=520
                )
            elif hasattr(px, 'density_mapbox'):
                fig_map = px.density_mapbox(
                    filtered_df, lat="lat", lon="lon", z="사고건수", radius=25,
                    center=dict(lat=center_lat, lon=center_lon), zoom=11.5,
                    mapbox_style="open-street-map", color_continuous_scale="YlOrRd", opacity=0.8, height=520
                )
            elif hasattr(go, 'Densitymap'):
                fig_map = go.Figure(go.Densitymap(
                    lat=filtered_df['lat'], lon=filtered_df['lon'], z=filtered_df['사고건수'],
                    radius=25, colorscale="YlOrRd"
                ))
            else:
                fig_map = go.Figure(go.Densitymapbox(
                    lat=filtered_df['lat'], lon=filtered_df['lon'], z=filtered_df['사고건수'],
                    radius=25, colorscale="YlOrRd"
                ))
        else:
            if hasattr(px, 'scatter_map'):
                fig_map = px.scatter_map(
                    filtered_df, lat="lat", lon="lon", hover_name="사고유형",
                    hover_data=["사고내용", "피해운전자 차종", "법규위반"], color="사고내용",
                    color_discrete_map={"사망": "red", "중상": "orange", "경상": "blue", "부상신고": "green"},
                    center=dict(lat=center_lat, lon=center_lon), zoom=11.5, height=520, map_style="open-street-map"
                )
            else:
                fig_map = px.scatter_mapbox(
                    filtered_df, lat="lat", lon="lon", hover_name="사고유형",
                    hover_data=["사고내용", "피해운전자 차종", "법규위반"], color="사고내용",
                    color_discrete_map={"사망": "red", "중상": "orange", "경상": "blue", "부상신고": "green"},
                    center=dict(lat=center_lat, lon=center_lon), zoom=11.5, height=520, mapbox_style="open-street-map"
                )
            fig_map.update_traces(marker=dict(size=9))

        if selected_fac != "(선택 없음)" and selected_fac in facility_dfs:
            df_fac_sub = facility_dfs[selected_fac]
            
            if hasattr(go, 'Scattermap'):
                fig_map.add_trace(go.Scattermap(
                    lat=df_fac_sub['lat'], lon=df_fac_sub['lon'], mode='markers',
                    marker=dict(size=7, color='cyan', opacity=0.9),
                    name=f"시설물: {selected_fac} ({len(df_fac_sub)}개소)", hoverinfo='text', text=selected_fac
                ))
            elif hasattr(go, 'Scattermapbox'):
                fig_map.add_trace(go.Scattermapbox(
                    lat=df_fac_sub['lat'], lon=df_fac_sub['lon'], mode='markers',
                    marker=dict(size=7, color='cyan', opacity=0.9),
                    name=f"시설물: {selected_fac} ({len(df_fac_sub)}개소)", hoverinfo='text', text=selected_fac
                ))

        fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, width='stretch')
    else:
        st.warning("선택한 조건에 해당하는 실제 사고 지점이 없습니다.")

# --- TAB 3: 반경 300m 도로안전 시설물 감사 (Safety Audit) ---
with tab3:
    st.subheader("🔍 사고 중심 반경 300m 도로안전 시설물 감사 (Safety Audit)")
    st.caption("선택한 기상/주야 조건 사고 지점 주변의 19종 도로시설물 현황을 자동 계산하여 안전 시설 공백을 평가합니다.")
    
    filtered_df = df_raw[(df_raw['기상상태'] == weather) & (df_raw['주야'] == day_night)].copy()
    
    if len(filtered_df) > 0 and len(facility_dfs) > 0:
        c_lat = filtered_df['lat'].mean()
        c_lon = filtered_df['lon'].mean()
        
        audit_results = {}
        for fac_name, df_fac in facility_dfs.items():
            if not df_fac.empty:
                d_lat = (df_fac['lat'] - c_lat) * 111000
                d_lon = (df_fac['lon'] - c_lon) * 88000
                dist = np.sqrt(d_lat**2 + d_lon**2)
                audit_results[fac_name] = int((dist <= 300).sum())
            else:
                audit_results[fac_name] = 0
                
        df_audit = pd.DataFrame(list(audit_results.items()), columns=["도로시설물", "반경 300m 내 개수"]).sort_values("반경 300m 내 개수", ascending=False)
        
        col_au1, col_au2 = st.columns([1.2, 1])
        with col_au1:
            fig_audit = px.bar(
                df_audit, x="반경 300m 내 개수", y="도로시설물", orientation="h",
                color="반경 300m 내 개수", color_continuous_scale="Viridis",
                title="사고 밀집 구역 주변 19종 도로시설물 설치 현황 (개소)"
            )
            fig_audit.update_layout(height=450)
            st.plotly_chart(fig_audit, width='stretch')
            
        with col_au2:
            st.write("##### 📋 안전시설 공백 진단 결과")
            st.write(f"- **분석 기준점 (중심 좌표):** 위도 {c_lat:.4f}, 경도 {c_lon:.4f}")
            st.write(f"- **주요 시설 설치 현황:**")
            
            key_items = ['신호등', '횡단보도', '과속방지턱', '미끄럼방지시설', '보안등', '보호구역']
            for item in key_items:
                cnt = audit_results.get(item, 0)
                status_icon = "✅" if cnt > 0 else "⚠️ (보완 필요)"
                st.write(f"  • **{item}:** {cnt}개 {status_icon}")
                
            if audit_results.get('미끄럼방지시설', 0) == 0 and weather in ['비', '눈']:
                st.warning("⚠️ **악천후 고위험 구간:** 주변 300m 이내 미끄럼방지시설이 부족합니다.")
            elif audit_results.get('보안등', 0) < 5 and day_night == '야간':
                st.warning("⚠️ **야간 조명 취약 구간:** 주변 보안등/가로등 추가 설치를 권장합니다.")
            else:
                st.success("✅ 기본 안전시설 배치가 확인되었습니다.")
    else:
        st.info("시설물 감사 데이터 계산을 준비 중입니다.")

# --- TAB 4: 4사분면 분석 ---
with tab4:
    st.subheader("📊 가해➔피해 차종 조합 4사분면 종합 평가")
    quad_data = pd.DataFrame([
        {"조합": "승용→보행자", "건수": 403, "중상률": 46.6, "사분면": "1사분면 (고빈도·고위험)"},
        {"조합": "승용→이륜", "건수": 187, "중상률": 48.7, "사분면": "1사분면 (고빈도·고위험)"},
        {"조합": "승용→자전거", "건수": 78, "중상률": 42.3, "사분면": "1사분면 (고빈도·고위험)"},
        {"조합": "화물→보행자", "건수": 77, "중상률": 46.8, "사분면": "1사분면 (고빈도·고위험)"},
        {"조합": "이륜→보행자", "건수": 31, "중상률": 45.2, "사분면": "2사분면 (저빈도·고위험)"},
        {"조합": "승용→원동기", "건수": 26, "중상률": 53.8, "사분면": "2사분면 (저빈도·고위험)"},
        {"조합": "승합→보행자", "건수": 20, "중상률": 50.0, "사분면": "2사분면 (저빈도·고위험)"},
        {"조합": "승합→승용", "건수": 42, "중상률": 11.9, "사분면": "3사분면 (저빈도·저위험)"},
        {"조합": "기타→승용", "건수": 36, "중상률": 16.7, "사분면": "3사분면 (저빈도·저위험)"},
        {"조합": "화물→화물", "건수": 33, "중상률": 12.1, "사분면": "3사분면 (저빈도·저위험)"},
        {"조합": "이륜→승용", "건수": 32, "중상률": 6.3, "사분면": "3사분면 (저빈도·저위험)"},
        {"조합": "승용→승용", "건수": 1243, "중상률": 8.2, "사분면": "4사분면 (고빈도·저위험)"},
        {"조합": "화물→승용", "건수": 180, "중상률": 11.1, "사분면": "4사분면 (고빈도·저위험)"},
        {"조합": "승용→화물", "건수": 93, "중상률": 15.1, "사분면": "4사분면 (고빈도·저위험)"},
        {"조합": "승용→승합", "건수": 44, "중상률": 2.3, "사분면": "4사분면 (고빈도·저위험)"},
    ])
    
    col_q1, col_q2 = st.columns([1.5, 1])
    with col_q1:
        fig_bubble = px.scatter(
            quad_data, x="건수", y="중상률", size="건수", color="사분면", hover_name="조합",
            hover_data={"건수": True, "중상률": ":.1f%", "사분면": True}, log_x=True,
            labels={"건수": "사고 건수 (Log Scale)", "중상률": "중상 이상률 (%)"},
            color_discrete_map={
                "1사분면 (고빈도·고위험)": "#D9534F",
                "2사분면 (저빈도·고위험)": "#F0AD4E",
                "3사분면 (저빈도·저위험)": "#5BC0DE",
                "4사분면 (고빈도·저위험)": "#0275D8"
            }
        )
        fig_bubble.add_hline(y=21.9, line_dash="dash", line_color="red", annotation_text="평균 중상률 (21.9%)")
        fig_bubble.add_vline(x=44, line_dash="dash", line_color="gray", annotation_text="건수 중앙값 (44건)")
        fig_bubble.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_bubble, width='stretch')

    with col_q2:
        st.write("##### AI 피처 중요도 (Random Forest)")
        imp_df = pd.DataFrame({
            '요인': ['피해자 차종', '피해자 연령대', '사고 유형', '도로 형태', '기상 상태', '노면 상태', '주야 구분'],
            '중요도': [0.43, 0.26, 0.15, 0.08, 0.04, 0.02, 0.02]
        }).sort_values('중요도', ascending=True)
        fig_imp = px.bar(imp_df, x='중요도', y='요인', orientation='h', color='중요도', color_continuous_scale='Blues')
        fig_imp.update_layout(height=380, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_imp, width='stretch')

# --- TAB 5: 강릉시 취약 유형별 사고 추이 (강릉 원본 데이터 기반 추출) ---
with tab5:
    st.subheader("📈 강릉시 5대 취약 사고 유형별 추이 분석")
    st.caption("강릉시 원본 사고 데이터(2021~2024년)에서 직접 추출·집계한 강릉시 전용 통계입니다.")
    
    if not df_special.empty:
        col_sp1, col_sp2 = st.columns([1.2, 1])
        
        with col_sp1:
            st.write("##### 1. 강릉시 취약 유형별 연도별 사고 건수 추이")
            fig_line = px.line(
                df_special, x="연도", y="사고(건)", color="유형", markers=True,
                labels={"사고(건)": "사고 건수", "연도": "연도"},
                title="강릉시 특수 유형별 사고 발생 건수 변화"
            )
            fig_line.update_layout(height=380)
            st.plotly_chart(fig_line, width='stretch')

        with col_sp2:
            st.write("##### 2. 최근 연도 기준 강릉시 사고 사망자 수 비중")
            latest_year = df_special['연도'].max()
            df_latest = df_special[df_special['연도'] == latest_year]
            
            fig_pie = px.pie(
                df_latest, names="유형", values="사망(명)",
                title=f"{latest_year} 강릉시 사고 유형별 사망자 비중",
                hole=0.4
            )
            fig_pie.update_layout(height=380)
            st.plotly_chart(fig_pie, width='stretch')

        st.markdown("---")
        st.write("##### 3. 강릉시 통합 데이터 요약 표")
        st.dataframe(df_special, width='stretch')
    else:
        st.info("강릉시 사고 데이터 분석 결과를 읽어오는 중입니다...")