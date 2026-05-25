import os
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib

# =========================================================
# 모델 불러오기 (스크립트 위치 기준 절대경로 → 실행 폴더 무관)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tomato_model.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="착과율 예측",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =========================================================
# 전역 스타일 (폰트 / 배경 / 위젯 디자인 / 애니메이션)
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=Outfit:wght@300;400;500;600;700&display=swap');

    :root{
        --tomato:#e8412b;
        --tomato-deep:#b21f12;
        --leaf:#3fa34d;
        --leaf-deep:#1f6b32;
        --cream:#fff4e9;
        --ink:#241410;
        --glass:rgba(255,244,233,.07);
        --glass-line:rgba(255,244,233,.16);
    }

    /* 배경: 따뜻한 다크 + 움직이는 그라데이션 메시 + 그레인 */
    .stApp{
        background:
            radial-gradient(900px 600px at 12% -8%, rgba(232,65,43,.28), transparent 60%),
            radial-gradient(800px 700px at 110% 10%, rgba(63,163,77,.22), transparent 55%),
            radial-gradient(700px 700px at 50% 120%, rgba(178,31,18,.30), transparent 60%),
            #160b08;
        background-attachment:fixed;
        color:var(--cream);
        font-family:'Outfit',sans-serif;
    }
    .stApp::before{
        content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.05;
        background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    }
    [data-testid="stHeader"]{background:transparent;}
    .block-container{padding-top:3.2rem;max-width:760px;position:relative;z-index:1;}

    /* 타이포 */
    h1,h2,h3{font-family:'Fraunces',serif!important;letter-spacing:-.5px;}
    .stApp p, .stApp label{font-family:'Outfit',sans-serif;}

    /* 카드(입력 영역 등) */
    .card{
        background:var(--glass);
        border:1px solid var(--glass-line);
        border-radius:22px;
        padding:26px 26px 8px;
        backdrop-filter:blur(14px);
        box-shadow:0 24px 60px -28px rgba(0,0,0,.7), inset 0 1px 0 rgba(255,255,255,.06);
        animation:rise .7s cubic-bezier(.2,.8,.2,1) both;
    }

    /* 숫자 입력 위젯 */
    [data-testid="stNumberInput"] label{
        color:var(--cream)!important;font-weight:600;font-size:.95rem;opacity:.9;
    }
    [data-testid="stNumberInput"] input{
        background:rgba(255,244,233,.06)!important;
        color:var(--cream)!important;
        border:1px solid var(--glass-line)!important;
        border-radius:14px!important;
        transition:border-color .25s ease, box-shadow .25s ease, transform .15s ease;
    }
    [data-testid="stNumberInput"] input:focus{
        border-color:var(--tomato)!important;
        box-shadow:0 0 0 4px rgba(232,65,43,.22)!important;
    }
    [data-testid="stNumberInput"] button{
        background:rgba(255,244,233,.06)!important;border:1px solid var(--glass-line)!important;
        color:var(--cream)!important;transition:background .2s ease;
    }
    [data-testid="stNumberInput"] button:hover{background:rgba(232,65,43,.35)!important;}

    /* 버튼 (공통) */
    .stButton>button{
        font-family:'Outfit',sans-serif;font-weight:700;
        border:none;border-radius:16px;padding:.75rem 1.2rem;color:#fff;
        background:linear-gradient(135deg,var(--tomato),var(--tomato-deep));
        box-shadow:0 14px 30px -12px rgba(232,65,43,.8);
        transition:transform .18s cubic-bezier(.2,.8,.2,1), box-shadow .18s ease, filter .18s ease;
    }
    .stButton>button:hover{transform:translateY(-3px) scale(1.02);filter:brightness(1.07);
        box-shadow:0 22px 40px -14px rgba(232,65,43,.9);}
    .stButton>button:active{transform:translateY(0) scale(.97);}

    /* 결과 박스 */
    [data-testid="stAlert"]{
        background:linear-gradient(135deg,rgba(63,163,77,.22),rgba(31,107,50,.18))!important;
        border:1px solid rgba(63,163,77,.4)!important;border-radius:18px!important;
        color:var(--cream)!important;animation:pop .5s cubic-bezier(.2,1.4,.4,1) both;
    }

    /* 데이터프레임 라운드 */
    [data-testid="stDataFrame"]{border-radius:16px;overflow:hidden;border:1px solid var(--glass-line);}

    @keyframes rise{from{opacity:0;transform:translateY(26px)}to{opacity:1;transform:none}}
    @keyframes pop{from{opacity:0;transform:scale(.9)}to{opacity:1;transform:scale(1)}}
    @keyframes fadein{from{opacity:0}to{opacity:1}}

    .stagger>*{animation:rise .6s cubic-bezier(.2,.8,.2,1) both;}
    .stagger>*:nth-child(1){animation-delay:.05s}
    .stagger>*:nth-child(2){animation-delay:.15s}
    .stagger>*:nth-child(3){animation-delay:.25s}
    .stagger>*:nth-child(4){animation-delay:.35s}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 상태 초기화
# =========================================================
if "entered" not in st.session_state:
    st.session_state.entered = False


# =========================================================
# 인트로 화면 : 큰 토마토(누르면 입장)
# =========================================================
def intro_screen():
    components.html(
        """
        <div id="wrap">
          <div class="halo"></div>
          <div id="tomato" title="눌러서 입장">
            <div class="leaf"></div>
            <div class="shine"></div>
            <div class="dot"></div>
          </div>
          <div class="hint">🍅 토마토를 누르면 입장합니다</div>
        </div>
        <style>
          *{box-sizing:border-box;margin:0;padding:0;}
          #wrap{height:430px;display:flex;flex-direction:column;align-items:center;
                justify-content:center;gap:26px;font-family:'Outfit',sans-serif;}
          .halo{position:absolute;width:360px;height:360px;border-radius:50%;
                background:radial-gradient(circle,rgba(232,65,43,.45),transparent 65%);
                filter:blur(10px);animation:breathe 3.4s ease-in-out infinite;}
          #tomato{position:relative;width:190px;height:175px;border-radius:48% 48% 50% 50%;
                background:radial-gradient(circle at 34% 28%,#ff6a4d,#e8412b 45%,#b21f12 100%);
                box-shadow:0 26px 50px -16px rgba(178,31,18,.8),inset -14px -16px 34px rgba(120,10,0,.55),
                           inset 12px 12px 26px rgba(255,180,150,.35);
                cursor:pointer;animation:bob 2.6s ease-in-out infinite;
                transition:transform .22s cubic-bezier(.2,.9,.3,1.2);}
          #tomato:hover{transform:scale(1.07) rotate(-2deg);}
          #tomato:active{transform:scale(.9);}
          .leaf{position:absolute;top:-16px;left:50%;transform:translateX(-50%);width:74px;height:34px;
                background:radial-gradient(circle at 50% 120%,#3fa34d,#1f6b32);
                clip-path:polygon(50% 0,68% 38%,100% 38%,74% 62%,86% 100%,50% 74%,14% 100%,26% 62%,0 38%,32% 38%);
                filter:drop-shadow(0 4px 4px rgba(0,0,0,.35));}
          .shine{position:absolute;top:24px;left:34px;width:46px;height:34px;border-radius:50%;
                background:rgba(255,240,225,.55);filter:blur(5px);transform:rotate(-25deg);}
          .dot{position:absolute;bottom:18px;left:50%;transform:translateX(-50%);width:10px;height:10px;
                border-radius:50%;background:rgba(120,10,0,.4);}
          .hint{color:#ffd9c9;font-weight:600;letter-spacing:.4px;opacity:.9;
                animation:pulse 2s ease-in-out infinite;}
          .seed{position:absolute;left:50%;top:42%;width:9px;height:13px;border-radius:50% 50% 50% 50%/60% 60% 40% 40%;
                background:#ffe6b0;opacity:0;}
          @keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-12px)}}
          @keyframes breathe{0%,100%{transform:scale(1);opacity:.7}50%{transform:scale(1.12);opacity:1}}
          @keyframes pulse{0%,100%{opacity:.55}50%{opacity:1}}
          @keyframes burst{to{opacity:0;transform:translate(var(--tx),var(--ty)) scale(.3) rotate(var(--r))}}
          .exploding #tomato{animation:none;transform:scale(1.25);opacity:0;transition:all .45s ease;}
          .exploding .halo{transform:scale(2.2);opacity:0;transition:all .45s ease;}
        </style>
        <script>
          const tomato = document.getElementById('tomato');
          const wrap = document.getElementById('wrap');
          tomato.addEventListener('click', () => {
            // 씨앗 파편 터뜨리기
            for(let i=0;i<26;i++){
              const s=document.createElement('div');s.className='seed';
              const ang=Math.random()*Math.PI*2, dist=120+Math.random()*150;
              s.style.setProperty('--tx',Math.cos(ang)*dist+'px');
              s.style.setProperty('--ty',Math.sin(ang)*dist+'px');
              s.style.setProperty('--r',(Math.random()*720-360)+'deg');
              s.style.background = Math.random()>.5 ? '#ffe6b0' : '#e8412b';
              wrap.appendChild(s);
              s.style.opacity='1';
              s.style.animation='burst .6s cubic-bezier(.2,.7,.3,1) forwards';
            }
            wrap.classList.add('exploding');
            // 애니메이션이 보인 뒤 Streamlit 입장 신호 (쿼리 파라미터)
            setTimeout(()=>{
              const url=new URL(window.parent.location);
              url.searchParams.set('enter','1');
              window.parent.history.replaceState({}, '', url);
              window.parent.location.reload();
            }, 620);
          });
        </script>
        """,
        height=450,
    )

    st.markdown(
        "<h1 style='text-align:center;font-size:3rem;margin-top:-6px;'>착과율 예측</h1>"
        "<p style='text-align:center;opacity:.75;margin-top:-10px;'>환경 데이터로 토마토 착과율을 예측합니다</p>",
        unsafe_allow_html=True,
    )

    # JS가 막힐 경우(iframe 정책 등)를 대비한 백업 입장 버튼
    c1, c2, c3 = st.columns([1, 1.2, 1])
    with c2:
        if st.button("입장하기", use_container_width=True):
            st.session_state.entered = True
            st.rerun()


# 쿼리 파라미터로 들어온 입장 신호 처리
if st.query_params.get("enter") == "1":
    st.session_state.entered = True
    st.query_params.clear()


# =========================================================
# 메인 화면
# =========================================================
def main_screen():
    # 입장 직후 1회 "씨앗 폭발" 연출
    components.html(
        """
        <div id="fx"></div>
        <style>
        #fx{position:fixed;inset:0;pointer-events:none;overflow:hidden;}
        .p{position:absolute;left:50%;top:30%;width:10px;height:14px;border-radius:50%;}
        @keyframes fly{to{transform:translate(var(--x),var(--y)) rotate(var(--r));opacity:0}}
        </style>
        <script>
        const fx=document.getElementById('fx');
        for(let i=0;i<40;i++){
          const p=document.createElement('div');p.className='p';
          const ang=Math.random()*Math.PI*2,d=200+Math.random()*420;
          p.style.setProperty('--x',Math.cos(ang)*d+'px');
          p.style.setProperty('--y',Math.sin(ang)*d+'px');
          p.style.setProperty('--r',(Math.random()*720)+'deg');
          p.style.background=['#e8412b','#ffe6b0','#3fa34d','#ff6a4d'][i%4];
          p.style.animation='fly '+(.7+Math.random()*.5)+'s cubic-bezier(.15,.7,.3,1) forwards';
          fx.appendChild(p);
        }
        setTimeout(()=>fx.remove(),1400);
        </script>
        """,
        height=0,
    )

    st.markdown(
        "<h1 style='font-size:2.6rem;'>🍅 착과율 예측 시스템</h1>"
        "<p style='opacity:.78;margin-top:-8px;'>내부온도 · 내부습도 · 지온을 입력하고 착과율을 예측해 보세요.</p>",
        unsafe_allow_html=True,
    )

    # 모델 로드 (실패 시 친절히 안내)
    try:
        rf_model = load_model()
    except FileNotFoundError:
        st.error(
            f"모델 파일을 찾을 수 없습니다.\n\n`tomato_model.pkl` 을 이 위치에 두세요:\n\n`{MODEL_PATH}`"
        )
        st.stop()
    except Exception as e:
        st.error(f"모델을 불러오는 중 오류가 발생했습니다: {type(e).__name__} — {e}")
        st.stop()

    st.markdown("<div class='card stagger'>", unsafe_allow_html=True)
    st.subheader("환경 정보 입력")

    col1, col2 = st.columns(2)
    with col1:
        temp = st.number_input("내부온도 (℃)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        soil_temp = st.number_input("지온 (℃)", min_value=0.0, max_value=50.0, value=20.0, step=0.1)
    with col2:
        humidity = st.number_input("내부습도 (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)

    predict = st.button("🍅 예측하기", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if predict:
        input_data = pd.DataFrame(
            [[temp, humidity, soil_temp]],
            columns=["내부온도", "내부습도", "지온"],
        )
        predicted = float(rf_model.predict(input_data)[0])
        value = max(0.0, min(100.0, predicted))  # 게이지용 0~100 클램프

        # 애니메이션 원형 게이지 + 카운트업
        components.html(
            f"""
            <div class="gw">
              <div class="ring" id="ring">
                <div class="inner">
                  <div class="num" id="num">0.0%</div>
                  <div class="lab">예측 착과율</div>
                </div>
              </div>
            </div>
            <style>
              @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;900&family=Outfit:wght@500&display=swap');
              .gw{{display:flex;justify-content:center;padding:14px 0 4px;font-family:'Outfit',sans-serif;}}
              .ring{{width:210px;height:210px;border-radius:50%;display:grid;place-items:center;
                    background:conic-gradient(#3fa34d 0deg, #e8412b 0deg, rgba(255,255,255,.08) 0deg);
                    transition:background 1.1s cubic-bezier(.2,.8,.2,1);
                    box-shadow:0 18px 45px -18px rgba(0,0,0,.6);}}
              .inner{{width:166px;height:166px;border-radius:50%;background:#1b0d09;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;
                    border:1px solid rgba(255,244,233,.12);}}
              .num{{font-family:'Fraunces',serif;font-size:2.5rem;font-weight:900;color:#fff4e9;}}
              .lab{{font-size:.85rem;letter-spacing:1px;color:#ffd9c9;opacity:.8;text-transform:uppercase;}}
            </style>
            <script>
              const target={value:.4f};
              const ring=document.getElementById('ring'), num=document.getElementById('num');
              requestAnimationFrame(()=>{{
                const deg=target/100*360;
                ring.style.background='conic-gradient(#3fa34d 0deg,#e8412b '+deg+'deg,rgba(255,255,255,.08) '+deg+'deg)';
              }});
              let cur=0;const t0=performance.now(),dur=1100;
              function step(t){{const p=Math.min(1,(t-t0)/dur);const e=1-Math.pow(1-p,3);
                num.textContent=(target*e).toFixed(1)+'%';if(p<1)requestAnimationFrame(step);}}
              requestAnimationFrame(step);
            </script>
            """,
            height=250,
        )

        st.success(f"예측 착과율: {predicted:.1f}%")

        # 입력값이 학습 가정 범위를 벗어났는지 가벼운 경고(외삽 위험)
        if not (10 <= temp <= 35) or not (40 <= humidity <= 90) or not (10 <= soil_temp <= 30):
            st.warning(
                "입력값이 일반적인 재배 환경 범위를 벗어났습니다. "
                "랜덤포레스트는 학습 범위 밖 예측(외삽)에 약하므로 결과 신뢰도가 낮을 수 있습니다."
            )

        with st.expander("입력 데이터 보기"):
            st.dataframe(input_data, use_container_width=True)


# =========================================================
# 라우팅
# =========================================================
if st.session_state.entered:
    main_screen()
else:
    intro_screen()
