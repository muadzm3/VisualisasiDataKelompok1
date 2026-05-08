import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import Counter
import io

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Pasar Kerja & Gaji Data Science",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp {
    background: #0D1117;
}

.main-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f172a 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,136,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    color: #f0f6ff;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #64748b;
    font-size: 1rem;
    margin: 0;
    font-weight: 400;
}
.badge {
    display: inline-block;
    background: rgba(0,136,255,0.15);
    color: #60a5fa;
    border: 1px solid rgba(0,136,255,0.3);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 1rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.metric-card {
    background: #131929;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #0088ff; }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #60a5fa;
    line-height: 1;
}
.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.4rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 1.3rem;
    color: #e2e8f0;
    margin: 0 0 0.3rem 0;
}
.section-sub {
    color: #475569;
    font-size: 0.85rem;
    margin-bottom: 1.2rem;
}
.chart-card {
    background: #0f1825;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
}
.insight-box {
    background: rgba(0,136,255,0.06);
    border-left: 3px solid #0088ff;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin-top: 1rem;
    font-size: 0.84rem;
    color: #94a3b8;
    line-height: 1.6;
}
.insight-box strong { color: #60a5fa; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0f1a !important;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
.sidebar-title {
    font-family: 'Space Mono', monospace;
    color: #0088ff;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid #1e293b;
}
.tab-label {
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    jobs = pd.read_csv("job_dataset.csv")
    sal = pd.read_csv("ds_salaries.csv", index_col=0)

    # Normalize experience level in jobs
    exp_map = {
        'Fresher': 'Fresher', 'Entry-Level': 'Entry-Level',
        'Junior': 'Junior', 'Mid-Level': 'Mid-Level',
        'Mid-level': 'Mid-Level', 'Mid-Senior': 'Mid-Senior',
        'Mid-Senior Level': 'Mid-Senior', 'Senior': 'Senior',
        'Senior-Level': 'Senior', 'Experienced': 'Experienced',
        'Lead': 'Lead'
    }
    jobs['ExperienceLevel'] = jobs['ExperienceLevel'].map(exp_map).fillna(jobs['ExperienceLevel'])

    # Decode salary experience level
    sal_exp = {'EN': 'Entry-Level', 'MI': 'Mid-Level', 'SE': 'Senior', 'EX': 'Executive'}
    sal['experience_label'] = sal['experience_level'].map(sal_exp)
    sal_size = {'S': 'Kecil', 'M': 'Menengah', 'L': 'Besar'}
    sal['company_size_label'] = sal['company_size'].map(sal_size)
    sal['remote_label'] = sal['remote_ratio'].map({0: 'On-site (0%)', 50: 'Hybrid (50%)', 100: 'Remote (100%)'})

    return jobs, sal

jobs_df, sal_df = load_data()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙ Filter Data</div>', unsafe_allow_html=True)

    # Salary filters
    st.markdown("**Dataset Gaji**")
    year_opts = sorted(sal_df['work_year'].unique())
    sel_years = st.multiselect("Tahun", year_opts, default=year_opts, key="years")

    exp_opts = sal_df['experience_label'].dropna().unique().tolist()
    sel_exp = st.multiselect("Level Pengalaman (Gaji)", exp_opts, default=exp_opts, key="sal_exp")

    sal_range = st.slider(
        "Rentang Gaji (USD)",
        int(sal_df['salary_in_usd'].min()),
        int(sal_df['salary_in_usd'].max()),
        (int(sal_df['salary_in_usd'].min()), int(sal_df['salary_in_usd'].max())),
        step=5000
    )

    st.markdown("---")
    st.markdown("**Dataset Lowongan Kerja**")
    job_exp_opts = sorted(jobs_df['ExperienceLevel'].unique())
    sel_job_exp = st.multiselect("Level Pengalaman (Lowongan)", job_exp_opts, default=job_exp_opts, key="job_exp")

    st.markdown("---")
    st.markdown('<div style="font-size:0.72rem;color:#334155;text-align:center;">📁 job_dataset.csv · ds_salaries.csv<br>Dashboard Penelitian Ilmiah</div>', unsafe_allow_html=True)

# ─── FILTER DATA ──────────────────────────────────────────────────────────────
filtered_sal = sal_df[
    (sal_df['work_year'].isin(sel_years)) &
    (sal_df['experience_label'].isin(sel_exp)) &
    (sal_df['salary_in_usd'] >= sal_range[0]) &
    (sal_df['salary_in_usd'] <= sal_range[1])
]
filtered_jobs = jobs_df[jobs_df['ExperienceLevel'].isin(sel_job_exp)]

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="badge">📊 Artikel Ilmiah · Visualisasi Data</div>
    <h1>Analisis Pasar Kerja & Gaji Data Science</h1>
    <p>Dashboard interaktif untuk eksplorasi tren gaji, distribusi skill, dan dinamika pasar kerja bidang teknologi & data science secara global.</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI METRICS ──────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
metrics = [
    (f"${filtered_sal['salary_in_usd'].mean():,.0f}", "Rata-rata Gaji (USD)"),
    (f"{len(filtered_sal):,}", "Data Gaji"),
    (f"{len(filtered_jobs):,}", "Lowongan Kerja"),
    (f"{filtered_jobs['Title'].nunique()}", "Jenis Pekerjaan"),
    (f"{filtered_sal['company_location'].nunique()}", "Negara"),
]
for col, (val, lbl) in zip([col1,col2,col3,col4,col5], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Tren Gaji Tahunan",
    "💰 Gaji per Experience",
    "🔥 Top Job Titles",
    "☁️ Skill Wordcloud",
    "🗺️ Heatmap Remote × Perusahaan"
])

PLOT_BG = "#0f1825"
PAPER_BG = "#0f1825"
FONT_COLOR = "#94a3b8"
GRID_COLOR = "#1e293b"
ACCENT = ["#0088ff", "#06b6d4", "#8b5cf6", "#f59e0b", "#10b981", "#ef4444"]

def base_layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#e2e8f0", family="Plus Jakarta Sans"), x=0),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=FONT_COLOR, family="Plus Jakarta Sans"),
        height=height,
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e293b", font=dict(size=11)),
        xaxis=dict(gridcolor=GRID_COLOR, linecolor="#1e293b", tickfont=dict(size=11)),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor="#1e293b", tickfont=dict(size=11)),
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TREN GAJI TAHUNAN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📈 Tren Rata-rata Gaji Data Science per Tahun</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Perubahan kompensasi dari 2020–2022 berdasarkan level pengalaman</p>', unsafe_allow_html=True)

    trend = filtered_sal.groupby(['work_year', 'experience_label'])['salary_in_usd'].mean().reset_index()
    trend.columns = ['Tahun', 'Level', 'Rata-rata Gaji (USD)']

    fig1 = go.Figure()
    for i, lvl in enumerate(trend['Level'].unique()):
        d = trend[trend['Level'] == lvl]
        fig1.add_trace(go.Scatter(
            x=d['Tahun'], y=d['Rata-rata Gaji (USD)'],
            mode='lines+markers',
            name=lvl,
            line=dict(color=ACCENT[i % len(ACCENT)], width=2.5),
            marker=dict(size=8, symbol='circle'),
            hovertemplate=f"<b>{lvl}</b><br>Tahun: %{{x}}<br>Gaji: $%{{y:,.0f}}<extra></extra>"
        ))

    # Overall average line
    overall = filtered_sal.groupby('work_year')['salary_in_usd'].mean().reset_index()
    fig1.add_trace(go.Scatter(
        x=overall['work_year'], y=overall['salary_in_usd'],
        mode='lines',
        name='Rata-rata Keseluruhan',
        line=dict(color='#334155', width=1.5, dash='dot'),
        hovertemplate="Rata-rata: $%{y:,.0f}<extra></extra>"
    ))

    base_layout(fig1, height=430)
    fig1.update_xaxes(tickvals=sorted(filtered_sal['work_year'].unique()), tickformat='d')
    fig1.update_yaxes(tickprefix='$', tickformat=',')
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <strong>Temuan:</strong> Rata-rata gaji Data Science menunjukkan pertumbuhan konsisten dari 2020 ke 2022.
    Level <strong>Executive (EX)</strong> mencatat kenaikan paling signifikan, mengindikasikan meningkatnya permintaan
    terhadap talenta senior di industri data. Posisi <strong>Entry-Level</strong> juga mengalami kenaikan,
    mencerminkan kompetisi rekrutmen yang semakin ketat.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — GAJI PER EXPERIENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">💰 Distribusi Gaji per Level Pengalaman</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Box plot & bar chart rata-rata gaji berdasarkan level karier</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        order = ['Entry-Level', 'Mid-Level', 'Senior', 'Executive']
        order_present = [o for o in order if o in filtered_sal['experience_label'].values]

        fig2a = go.Figure()
        for i, lvl in enumerate(order_present):
            d = filtered_sal[filtered_sal['experience_label'] == lvl]['salary_in_usd']
            fig2a.add_trace(go.Box(
                y=d, name=lvl,
                marker_color=ACCENT[i],
                line_color=ACCENT[i],
                fillcolor=f"rgba({int(ACCENT[i][1:3],16)},{int(ACCENT[i][3:5],16)},{int(ACCENT[i][5:7],16)},0.2)",
                boxmean='sd',
                hovertemplate=f"<b>{lvl}</b><br>Gaji: $%{{y:,.0f}}<extra></extra>"
            ))
        base_layout(fig2a, height=400)
        fig2a.update_yaxes(tickprefix='$', tickformat=',')
        st.plotly_chart(fig2a, use_container_width=True)

    with col_b:
        avg_sal = filtered_sal.groupby('experience_label')['salary_in_usd'].agg(['mean','median','count']).reset_index()
        avg_sal.columns = ['Level', 'Rata-rata', 'Median', 'Jumlah']
        avg_sal = avg_sal[avg_sal['Level'].isin(order_present)]
        avg_sal['Level'] = pd.Categorical(avg_sal['Level'], categories=order_present, ordered=True)
        avg_sal = avg_sal.sort_values('Level')

        fig2b = go.Figure(go.Bar(
            x=avg_sal['Rata-rata'],
            y=avg_sal['Level'],
            orientation='h',
            marker=dict(
                color=ACCENT[:len(avg_sal)],
                line=dict(width=0)
            ),
            text=[f"${v:,.0f}" for v in avg_sal['Rata-rata']],
            textposition='outside',
            textfont=dict(color='#94a3b8', size=11),
            hovertemplate="<b>%{y}</b><br>Rata-rata: $%{x:,.0f}<extra></extra>"
        ))
        base_layout(fig2b, height=400)
        fig2b.update_xaxes(tickprefix='$', tickformat=',', title='Rata-rata Gaji (USD)')
        st.plotly_chart(fig2b, use_container_width=True)

    # Stats table
    st.dataframe(
        avg_sal.style
            .format({'Rata-rata': '${:,.0f}', 'Median': '${:,.0f}'})
            .set_properties(**{'background-color': '#0f1825', 'color': '#94a3b8', 'border': '1px solid #1e293b'}),
        use_container_width=True, hide_index=True
    )

    st.markdown("""
    <div class="insight-box">
    💡 <strong>Temuan:</strong> Terdapat kesenjangan gaji yang signifikan antar level.
    Level <strong>Executive</strong> mendapatkan kompensasi ~3–4× lebih tinggi dari <strong>Entry-Level</strong>.
    Box plot menunjukkan variabilitas gaji yang besar pada level Senior, mengindikasikan
    perbedaan spesialisasi dan lokasi perusahaan yang memengaruhi kompensasi.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TOP JOB TITLES
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🔥 Top Job Titles Berdasarkan Jumlah Lowongan</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Posisi dengan permintaan pasar tertinggi di dataset lowongan kerja</p>', unsafe_allow_html=True)

    col_top = st.slider("Tampilkan Top N Pekerjaan", 5, 30, 15, key="top_n")

    title_counts = filtered_jobs['Title'].value_counts().head(col_top).reset_index()
    title_counts.columns = ['Judul Pekerjaan', 'Jumlah']
    title_counts = title_counts.sort_values('Jumlah', ascending=True)

    colors_grad = [
        f"rgba(0,{int(136 + (i / len(title_counts)) * 100)},{int(255 - (i / len(title_counts)) * 80)},0.85)"
        for i in range(len(title_counts))
    ]

    fig3 = go.Figure(go.Bar(
        x=title_counts['Jumlah'],
        y=title_counts['Judul Pekerjaan'],
        orientation='h',
        marker=dict(color=colors_grad, line=dict(width=0)),
        text=title_counts['Jumlah'],
        textposition='outside',
        textfont=dict(color='#64748b', size=11),
        hovertemplate="<b>%{y}</b><br>Jumlah Lowongan: %{x}<extra></extra>"
    ))

    base_layout(fig3, height=max(350, col_top * 28))
    fig3.update_xaxes(title='Jumlah Lowongan')
    fig3.update_yaxes(title='')
    st.plotly_chart(fig3, use_container_width=True)

    # Pie chart by experience for top jobs
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        exp_dist = filtered_jobs['ExperienceLevel'].value_counts().reset_index()
        exp_dist.columns = ['Level', 'Jumlah']
        fig3b = go.Figure(go.Pie(
            labels=exp_dist['Level'],
            values=exp_dist['Jumlah'],
            hole=0.5,
            marker=dict(colors=ACCENT[:len(exp_dist)], line=dict(color='#0f1825', width=2)),
            hovertemplate="<b>%{label}</b><br>%{value} lowongan<br>%{percent}<extra></extra>"
        ))
        fig3b.update_layout(
            title=dict(text="Distribusi Level Pengalaman", font=dict(color='#e2e8f0', size=13), x=0),
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font=dict(color=FONT_COLOR), height=300,
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig3b, use_container_width=True)

    with col_p2:
        st.markdown('<br><br>', unsafe_allow_html=True)
        top5 = filtered_jobs['Title'].value_counts().head(5)
        st.markdown("**Top 5 Pekerjaan Paling Dicari:**")
        for i, (title, count) in enumerate(top5.items()):
            pct = count / len(filtered_jobs) * 100
            st.markdown(f"""
            <div style="display:flex;align-items:center;margin-bottom:0.5rem;">
                <span style="font-family:Space Mono;color:{ACCENT[i]};font-size:0.85rem;width:24px;">{i+1}.</span>
                <div style="flex:1;background:#1e293b;border-radius:20px;height:6px;margin:0 10px;">
                    <div style="width:{pct*3:.0f}%;background:{ACCENT[i]};height:6px;border-radius:20px;"></div>
                </div>
                <span style="font-size:0.82rem;color:#94a3b8;min-width:100px;">{title}</span>
                <span style="font-family:Space Mono;color:{ACCENT[i]};font-size:0.8rem;margin-left:8px;">{count}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    💡 <strong>Temuan:</strong> Data Engineer, Data Scientist, dan Data Analyst mendominasi permintaan pasar.
    Hal ini mengindikasikan bahwa industri semakin bergantung pada infrastruktur data dan analisis prediktif.
    Posisi berbasis <strong>AI & Machine Learning</strong> juga menunjukkan tren pertumbuhan yang kuat.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — WORDCLOUD SKILL
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">☁️ Word Cloud Skill yang Paling Banyak Diminta</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Frekuensi kemunculan skill teknis & non-teknis di seluruh lowongan kerja</p>', unsafe_allow_html=True)

    col_wc1, col_wc2 = st.columns([1, 2])

    with col_wc1:
        wc_filter = st.radio("Tampilkan:", ["Semua Skill", "Skill Teknis", "Soft Skill"], key="wc_filter")
        min_freq = st.slider("Frekuensi minimum", 5, 50, 10, key="wc_freq")

    # Build skill counter
    skill_counter = Counter()
    for _, row in filtered_jobs.iterrows():
        skills = [s.strip() for s in str(row['Skills']).split(';')]
        skill_counter.update(skills)

    soft_skills = {'Communication', 'Leadership', 'Teamwork', 'Problem-solving', 'Collaboration',
                   'Project management', 'Adaptability', 'Team collaboration', 'Creativity',
                   'Time management', 'Critical thinking', 'Attention to detail', 'Multitasking',
                   'Interpersonal skills', 'Analytical thinking'}

    if wc_filter == "Skill Teknis":
        skill_counter = Counter({k: v for k, v in skill_counter.items() if k not in soft_skills})
    elif wc_filter == "Soft Skill":
        skill_counter = Counter({k: v for k, v in skill_counter.items() if k in soft_skills})

    skill_counter = Counter({k: v for k, v in skill_counter.items() if v >= min_freq})

    with col_wc2:
        if skill_counter:
            custom_colors = ['#0088ff', '#06b6d4', '#8b5cf6', '#10b981', '#f59e0b', '#3b82f6', '#a78bfa']
            def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                import random
                return random.choice(custom_colors)

            wc = WordCloud(
                width=700, height=380,
                background_color='#0f1825',
                colormap=None,
                color_func=color_func,
                max_words=80,
                prefer_horizontal=0.8,
                min_font_size=10,
                max_font_size=72,
                margin=8,
                font_path=None
            ).generate_from_frequencies(skill_counter)

            fig_wc, ax_wc = plt.subplots(figsize=(9, 4.5))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            fig_wc.patch.set_facecolor('#0f1825')
            ax_wc.set_facecolor('#0f1825')
            plt.tight_layout(pad=0)
            st.pyplot(fig_wc, use_container_width=True)
            plt.close()
        else:
            st.info("Tidak ada skill yang memenuhi kriteria filter.")

    # Top 15 skill bar
    top_skills = pd.DataFrame(skill_counter.most_common(15), columns=['Skill', 'Frekuensi'])

    fig4b = go.Figure(go.Bar(
        x=top_skills['Frekuensi'],
        y=top_skills['Skill'],
        orientation='h',
        marker=dict(
            color=top_skills['Frekuensi'],
            colorscale=[[0, '#1e3a5f'], [0.5, '#0088ff'], [1, '#06b6d4']],
            line=dict(width=0)
        ),
        text=top_skills['Frekuensi'],
        textposition='outside',
        textfont=dict(color='#64748b', size=10),
        hovertemplate="<b>%{y}</b><br>Frekuensi: %{x}<extra></extra>"
    ))
    top_skills_sorted = top_skills.sort_values('Frekuensi', ascending=True)
    fig4b = go.Figure(go.Bar(
        x=top_skills_sorted['Frekuensi'],
        y=top_skills_sorted['Skill'],
        orientation='h',
        marker=dict(
            color=top_skills_sorted['Frekuensi'],
            colorscale=[[0, '#1e293b'], [0.5, '#1e3a8a'], [1, '#0088ff']],
            line=dict(width=0)
        ),
        text=top_skills_sorted['Frekuensi'],
        textposition='outside',
        textfont=dict(color='#64748b', size=10),
        hovertemplate="<b>%{y}</b><br>Frekuensi: %{x}<extra></extra>"
    ))
    base_layout(fig4b, "Top 15 Skill Berdasarkan Frekuensi", height=380)
    st.plotly_chart(fig4b, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <strong>Temuan:</strong> <strong>Python</strong> mendominasi sebagai skill teknis paling dicari,
    diikuti oleh Git, Java, dan SQL. Di antara soft skill, <strong>Communication</strong> dan
    <strong>Leadership</strong> menjadi yang teratas — menandakan bahwa industri tidak hanya
    membutuhkan kompetensi teknis, tetapi juga kecakapan interpersonal yang kuat.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HEATMAP REMOTE × COMPANY SIZE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🗺️ Heatmap: Remote Ratio × Ukuran Perusahaan</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Rata-rata gaji dan distribusi karyawan berdasarkan skema kerja dan skala perusahaan</p>', unsafe_allow_html=True)

    col_h1, col_h2 = st.columns(2)

    with col_h1:
        # Heatmap avg salary
        heat_data = filtered_sal.groupby(['remote_label', 'company_size_label'])['salary_in_usd'].mean().unstack(fill_value=0)
        remote_order = ['On-site (0%)', 'Hybrid (50%)', 'Remote (100%)']
        size_order = ['Kecil', 'Menengah', 'Besar']
        heat_data = heat_data.reindex(index=[r for r in remote_order if r in heat_data.index],
                                       columns=[s for s in size_order if s in heat_data.columns])

        fig5a = go.Figure(go.Heatmap(
            z=heat_data.values,
            x=heat_data.columns.tolist(),
            y=heat_data.index.tolist(),
            colorscale=[[0,'#0a0f1a'],[0.3,'#0c2461'],[0.6,'#0088ff'],[1,'#06b6d4']],
            text=[[f"${v:,.0f}" for v in row] for row in heat_data.values],
            texttemplate="%{text}",
            textfont=dict(size=13, color='white'),
            hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Rata-rata: $%{z:,.0f}<extra></extra>",
            showscale=True,
            colorbar=dict(tickfont=dict(color='#64748b'), title=dict(text='USD', font=dict(color='#64748b')))
        ))
        base_layout(fig5a, "Rata-rata Gaji (USD)", height=320)
        fig5a.update_xaxes(title='Ukuran Perusahaan')
        fig5a.update_yaxes(title='Skema Kerja')
        st.plotly_chart(fig5a, use_container_width=True)

    with col_h2:
        # Heatmap count
        heat_count = filtered_sal.groupby(['remote_label', 'company_size_label']).size().unstack(fill_value=0)
        heat_count = heat_count.reindex(index=[r for r in remote_order if r in heat_count.index],
                                         columns=[s for s in size_order if s in heat_count.columns])

        fig5b = go.Figure(go.Heatmap(
            z=heat_count.values,
            x=heat_count.columns.tolist(),
            y=heat_count.index.tolist(),
            colorscale=[[0,'#0a0f1a'],[0.3,'#2d1b69'],[0.6,'#7c3aed'],[1,'#a78bfa']],
            text=heat_count.values,
            texttemplate="%{text}",
            textfont=dict(size=13, color='white'),
            hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Jumlah: %{z}<extra></extra>",
            showscale=True,
            colorbar=dict(tickfont=dict(color='#64748b'), title=dict(text='Karyawan', font=dict(color='#64748b')))
        ))
        base_layout(fig5b, "Jumlah Karyawan", height=320)
        fig5b.update_xaxes(title='Ukuran Perusahaan')
        fig5b.update_yaxes(title='Skema Kerja')
        st.plotly_chart(fig5b, use_container_width=True)

    # Breakdown bar
    remote_avg = filtered_sal.groupby('remote_label')['salary_in_usd'].mean().reset_index()
    remote_avg.columns = ['Skema Kerja', 'Rata-rata Gaji']
    remote_avg = remote_avg.sort_values('Rata-rata Gaji', ascending=False)

    fig5c = go.Figure()
    for i, row in remote_avg.iterrows():
        fig5c.add_trace(go.Bar(
            x=[row['Skema Kerja']], y=[row['Rata-rata Gaji']],
            name=row['Skema Kerja'],
            marker_color=ACCENT[list(remote_avg['Skema Kerja']).index(row['Skema Kerja'])],
            text=[f"${row['Rata-rata Gaji']:,.0f}"],
            textposition='outside',
            textfont=dict(color='#94a3b8'),
            hovertemplate=f"<b>{row['Skema Kerja']}</b><br>Rata-rata: $%{{y:,.0f}}<extra></extra>"
        ))

    base_layout(fig5c, "Rata-rata Gaji per Skema Kerja", height=300)
    fig5c.update_yaxes(tickprefix='$', tickformat=',')
    fig5c.update_layout(showlegend=False, bargap=0.4)
    st.plotly_chart(fig5c, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <strong>Temuan:</strong> Perusahaan berukuran <strong>menengah-besar</strong> dengan skema <strong>remote 100%</strong>
    cenderung menawarkan gaji lebih kompetitif. Pola ini konsisten dengan tren global pasca-pandemi
    di mana remote work mendorong akses ke talent pool global tanpa batasan geografis.
    Perusahaan kecil menunjukkan rentang gaji lebih sempit namun dengan fleksibilitas kerja yang variatif.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.78rem; padding:1rem 0; font-family:'Space Mono',monospace;">
    📊 Dashboard Visualisasi Data Ilmiah · Sumber: job_dataset.csv & ds_salaries.csv<br>
    Dibuat untuk keperluan artikel ilmiah · Teknologi: Streamlit · Plotly · WordCloud
</div>
""", unsafe_allow_html=True)
