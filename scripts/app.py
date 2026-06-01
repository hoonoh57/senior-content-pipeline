import streamlit as st, json, zipfile, io, shutil, os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
PROJECTS_DIR = ROOT / "projects"
TEMPLATES_DIR = ROOT / "_templates"
PROJECTS_DIR.mkdir(exist_ok=True)
STAGES = ["01_주제확정","02_시나리오","03_미디어","04_동영상","완료"]
VOX_URL = "http://127.0.0.1:7860/"
VOX_API = "/generate_tts_integrated"

def list_projects():
    items = []
    for d in sorted(PROJECTS_DIR.iterdir()):
        f = d / "project.json"
        if d.is_dir() and f.exists():
            try: items.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception: pass
    return items

def project_dir(pid): return PROJECTS_DIR / pid
def media_dir(pid):
    d = project_dir(pid)/"media"; d.mkdir(parents=True, exist_ok=True); return d
def audio_dir(pid):
    d = project_dir(pid)/"audio"; d.mkdir(parents=True, exist_ok=True); return d
def video_dir(pid):
    d = project_dir(pid)/"video"; d.mkdir(parents=True, exist_ok=True); return d

def save_project(meta):
    d = project_dir(meta["id"]); d.mkdir(parents=True, exist_ok=True)
    (d/"project.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def next_id():
    nums = []
    for d in PROJECTS_DIR.iterdir():
        if d.is_dir() and "-" in d.name:
            h = d.name.split("-",1)[0]
            if h.isdigit(): nums.append(int(h))
    return f"{(max(nums)+1) if nums else 1:04d}"

REQUIRED_TOP = ["final_title","script","self_eval","character_sheet","scenes","scene_count","coverage_note"]
REQUIRED_SCENE = ["scene","type","time_start","time_end","narration","image_prompt","image_filename"]

def validate_scenario(data):
    errs = []
    if not isinstance(data, dict): return ["최상위가 JSON 객체가 아닙니다."]
    for k in REQUIRED_TOP:
        if k not in data: errs.append(f"필수 필드 누락: {k}")
    se = data.get("self_eval", {})
    if isinstance(se, dict):
        a = se.get("average")
        if isinstance(a,(int,float)) and a<8: errs.append(f"자체평가 평균 {a} < 8")
    sc = data.get("scenes", [])
    if not isinstance(sc, list) or not sc: errs.append("scenes 배열 오류")
    else:
        for i,s in enumerate(sc,1):
            for k in REQUIRED_SCENE:
                if k not in s: errs.append(f"scene {i} 필드 누락: {k}")
            if s.get("type") not in ("character","infographic"): errs.append(f"scene {i} type 오류")
    return errs

def load_scenario(pid):
    f = project_dir(pid)/"scenario.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

def build_image_prompt(data):
    cs = data.get("character_sheet", {})
    L = ["# 이미지 생성 일괄 의뢰 (전역 참조 → 장면별)","# 지정 파일명으로 생성 후 ZIP 다운로드.","",
         "## STEP 1 - character.png 먼저","Character: "+cs.get("description",""),
         "Negative: "+cs.get("negative_prompt",""),"Aspect ratio: 16:9","",
         "## STEP 2 - 장면별 (인물 참조). character는 텍스트 없이, infographic은 시안만.",""]
    for s in data.get("scenes", []):
        L.append(f"[{s.get('image_filename','')}] scene {s.get('scene')} | {s.get('type')} | {s.get('time_label','')}")
        L.append("  "+("INFOGRAPHIC IDEA: "+s.get("infographic_idea","") if s.get("type")=="infographic" else "PROMPT: "+s.get("image_prompt","")))
        L.append("  Negative: "+cs.get("negative_prompt","")); L.append("")
    return "\n".join(L)

def expected_image_names(data):
    n = {"character.png"}
    for s in data.get("scenes", []): n.add(s.get("image_filename", f"scene_{s.get('scene'):02d}.png"))
    return n

def audio_name(no): return f"scene_{no:02d}.wav"

def _extract_audio_path(out):
    def ok(x): return isinstance(x,(str,os.PathLike)) and os.path.exists(x)
    if ok(out): return str(out)
    if isinstance(out, dict):
        for k in ("path","name","value"):
            if k in out and ok(out[k]): return str(out[k])
    if isinstance(out,(tuple,list)):
        for it in out:
            r=_extract_audio_path(it)
            if r: return r
    return None

def tts_generate(text, vcfg):
    from gradio_client import Client, handle_file
    c = Client(VOX_URL)
    out = c.predict(script=text, ref_audio_path=handle_file(vcfg["ref_audio_path"]),
                    gen_mode=vcfg["gen_mode"], emotion_prompt=vcfg["emotion_prompt"],
                    speed=float(vcfg["speed"]), timesteps=float(vcfg["timesteps"]),
                    guidance=float(vcfg["guidance"]), ref_transcript=vcfg.get("ref_transcript",""),
                    api_name=VOX_API)
    p = _extract_audio_path(out)
    if not p: raise RuntimeError(f"오디오 경로 못 찾음: {type(out)} -> {out!r}")
    return p

def build_video(pid, data, fps=24, size=(1920,1080)):
    from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
    mdir, adir, vdir = media_dir(pid), audio_dir(pid), video_dir(pid)
    clips = []
    for s in data.get("scenes", []):
        no = s.get("scene")
        img = mdir / s.get("image_filename", f"scene_{no:02d}.png")
        aud = adir / audio_name(no)
        if not img.exists(): raise RuntimeError(f"scene {no} 이미지 없음: {img.name}")
        if not aud.exists(): raise RuntimeError(f"scene {no} 음성 없음: {aud.name}")
        ac = AudioFileClip(str(aud))
        ic = (ImageClip(str(img)).with_duration(ac.duration)
              .resized(height=size[1]).with_audio(ac))
        clips.append(ic)
    final = concatenate_videoclips(clips, method="compose")
    out = vdir / "final.mp4"
    final.write_videofile(str(out), fps=fps, codec="libx264", audio_codec="aac")
    return out


st.set_page_config(page_title="시니어 콘텐츠 파이프라인", layout="wide")
st.title("시니어 콘텐츠 파이프라인")
tab_p, tab_s, tab_m, tab_a, tab_v = st.tabs(["프로젝트","시나리오","미디어","음성","동영상"])

with tab_p:
    with st.form("new", clear_on_submit=True):
        title = st.text_input("제목/메모")
        if st.form_submit_button("생성") and title.strip():
            pid = next_id()
            save_project({"id": f"{pid}-{title.strip().replace(' ','-')[:20]}","title":title.strip(),
                          "stage":STAGES[0],"created":datetime.now().strftime("%Y-%m-%d %H:%M")})
            st.success("생성"); st.rerun()
    st.subheader("프로젝트 목록")
    for p in list_projects():
        c1,c2 = st.columns([3,2]); c1.write(f"**{p['id']}** · {p.get('created','')}")
        ns = c2.selectbox("단계", STAGES, index=STAGES.index(p["stage"]), key=f"stg_{p['id']}", label_visibility="collapsed")
        if ns != p["stage"]: p["stage"]=ns; save_project(p); st.rerun()

with tab_s:
    pr = list_projects()
    if not pr: st.info("먼저 프로젝트를 생성하세요.")
    else:
        sel = st.selectbox("프로젝트 선택", [p["id"] for p in pr], key="s_sel")
        meta = next(p for p in pr if p["id"]==sel); st.caption(f"단계: {meta['stage']}")
        tpl = TEMPLATES_DIR/"content_request.md"
        with st.expander("① 작가 AI 의뢰 프롬프트"):
            st.code(tpl.read_text(encoding="utf-8") if tpl.exists() else "(없음)", language="markdown")
        st.subheader("② JSON 등록/교체")
        raw = st.text_area("JSON", height=200, key="s_raw")
        if st.button("검증 후 저장", key="s_save"):
            try: d2=json.loads(raw)
            except Exception as e: st.error(f"파싱 실패: {e}")
            else:
                er=validate_scenario(d2)
                if er: st.error("검증 실패:\n- "+"\n- ".join(er))
                else:
                    (project_dir(sel)/"scenario.json").write_text(json.dumps(d2,ensure_ascii=False,indent=2),encoding="utf-8")
                    meta["stage"]="02_시나리오"; save_project(meta); st.success("저장"); st.rerun()
        data=load_scenario(sel)
        if data:
            st.subheader("③ 저장된 시나리오"); st.write(f"**제목:** {data.get('final_title','')}")
            se=data.get("self_eval",{}); st.write(f"자체평가 {se.get('average','-')} · 장면 {data.get('scene_count','-')}")
            with st.expander("대본 전문"): st.write(data.get("script",""))
            if st.button("시나리오 삭제", key="s_del"):
                (project_dir(sel)/"scenario.json").unlink(missing_ok=True); meta["stage"]=STAGES[0]; save_project(meta); st.rerun()

with tab_m:
    pr=list_projects()
    if not pr: st.info("먼저 프로젝트를 생성하세요.")
    else:
        sel=st.selectbox("프로젝트 선택",[p["id"] for p in pr],key="m_sel")
        meta=next(p for p in pr if p["id"]==sel); data=load_scenario(sel)
        if not data: st.warning("시나리오 없음")
        else:
            mdir=media_dir(sel)
            st.subheader("① 이미지 일괄 프롬프트"); st.code(build_image_prompt(data),language="text")
            st.subheader("② ZIP 일괄 등록")
            zu=st.file_uploader("이미지 ZIP",type=["zip"],key="m_zip")
            if zu and st.button("압축 해제 후 등록",key="m_go"):
                exp=expected_image_names(data); sv,sk=[],[]
                try:
                    with zipfile.ZipFile(io.BytesIO(zu.getvalue())) as zf:
                        for inf in zf.infolist():
                            if inf.is_dir(): continue
                            b=Path(inf.filename).name
                            if b in exp: (mdir/b).write_bytes(zf.read(inf)); sv.append(b)
                            else: sk.append(b)
                except zipfile.BadZipFile: st.error("ZIP 오류")
                else:
                    st.success(f"등록 {len(sv)}개")
                    if sk: st.warning("건너뜀: "+", ".join(sk))
                    st.rerun()
            sc=data.get("scenes",[])
            dn=sum(1 for s in sc if (mdir/s.get("image_filename",f"scene_{s.get('scene'):02d}.png")).exists())
            st.progress(dn/len(sc) if sc else 0, text=f"이미지 {dn}/{len(sc)}")
            for s in sc:
                fn=s.get("image_filename",f"scene_{s.get('scene'):02d}.png"); fp=mdir/fn
                with st.expander(f"scene {s.get('scene')} {'OK' if fp.exists() else '-'}"):
                    up=st.file_uploader(f"{fn} 교체",type=["png","jpg","jpeg"],key=f"mi_{s.get('scene')}")
                    if up: fp.write_bytes(up.getvalue()); st.rerun()
                    if fp.exists(): st.image(str(fp),width=240)

with tab_a:
    pr=list_projects()
    if not pr: st.info("먼저 프로젝트를 생성하세요.")
    else:
        sel=st.selectbox("프로젝트 선택",[p["id"] for p in pr],key="a_sel")
        meta=next(p for p in pr if p["id"]==sel); data=load_scenario(sel)
        if not data: st.warning("시나리오 없음")
        else:
            adir=audio_dir(sel)
            st.subheader("① 음성 설정")
            vc=meta.get("voice_cfg",{})
            ref=st.text_input("참조 음성 경로",vc.get("ref_audio_path",""))
            reft=st.text_area("참조 대본(선택)",vc.get("ref_transcript",""),height=60)
            gm=st.text_input("gen_mode",vc.get("gen_mode","제어형 클로닝 (참조 오디오 기반)"))
            em=st.text_input("emotion_prompt",vc.get("emotion_prompt","normal"))
            c1,c2,c3=st.columns(3)
            sp=c1.number_input("speed",value=float(vc.get("speed",1.0)),step=0.1)
            ts=c2.number_input("timesteps",value=float(vc.get("timesteps",15)),step=1.0)
            gd=c3.number_input("guidance",value=float(vc.get("guidance",2.0)),step=0.5)
            if st.button("음성 설정 저장",key="a_save"):
                meta["voice_cfg"]={"ref_audio_path":ref,"ref_transcript":reft,"gen_mode":gm,"emotion_prompt":em,"speed":sp,"timesteps":ts,"guidance":gd}
                save_project(meta); st.success("저장"); st.rerun()
            vcfg=meta.get("voice_cfg"); ready=bool(vcfg and vcfg.get("ref_audio_path") and Path(vcfg["ref_audio_path"]).exists())
            if not ready: st.warning("참조 음성 경로 확인 필요")
            st.subheader("② 장면별 음성")
            sc=data.get("scenes",[])
            dn=sum(1 for s in sc if (adir/audio_name(s.get("scene"))).exists())
            st.progress(dn/len(sc) if sc else 0, text=f"음성 {dn}/{len(sc)}")
            x1,x2=st.columns(2)
            if x1.button("빈 장면만 일괄 녹음",key="a_empty",disabled=not ready):
                todo=[s for s in sc if not (adir/audio_name(s.get("scene"))).exists()]; pb=st.progress(0.0)
                for i,s in enumerate(todo,1):
                    try: shutil.copyfile(tts_generate(s.get("narration",""),vcfg), adir/audio_name(s.get("scene")))
                    except Exception as e: st.error(f"scene {s.get('scene')} 실패: {e}"); break
                    pb.progress(i/len(todo) if todo else 1.0)
                st.success("완료"); st.rerun()
            if x2.button("전체 재녹음",key="a_all",disabled=not ready):
                pb=st.progress(0.0)
                for i,s in enumerate(sc,1):
                    try: shutil.copyfile(tts_generate(s.get("narration",""),vcfg), adir/audio_name(s.get("scene")))
                    except Exception as e: st.error(f"scene {s.get('scene')} 실패: {e}"); break
                    pb.progress(i/len(sc))
                st.success("완료"); st.rerun()
            for s in sc:
                no=s.get("scene"); ap=adir/audio_name(no)
                with st.expander(f"scene {no} · {s.get('time_label','')} {'OK' if ap.exists() else '-'}"):
                    txt=st.text_area("내레이션",s.get("narration",""),key=f"an_{no}",height=70)
                    b1,b2=st.columns(2)
                    if b1.button("녹음/재녹음",key=f"ar_{no}",disabled=not ready):
                        try: shutil.copyfile(tts_generate(txt,vcfg),ap); st.success("완료"); st.rerun()
                        except Exception as e: st.error(f"실패: {e}")
                    if ap.exists():
                        if b2.button("삭제",key=f"ad_{no}"): ap.unlink(missing_ok=True); st.rerun()
                        st.audio(str(ap))

with tab_v:
    pr=list_projects()
    if not pr: st.info("먼저 프로젝트를 생성하세요.")
    else:
        sel=st.selectbox("프로젝트 선택",[p["id"] for p in pr],key="v_sel")
        meta=next(p for p in pr if p["id"]==sel); data=load_scenario(sel)
        if not data: st.warning("시나리오 없음")
        else:
            mdir,adir,vdir=media_dir(sel),audio_dir(sel),video_dir(sel)
            sc=data.get("scenes",[])
            img_ok=sum(1 for s in sc if (mdir/s.get("image_filename",f"scene_{s.get('scene'):02d}.png")).exists())
            aud_ok=sum(1 for s in sc if (adir/audio_name(s.get("scene"))).exists())
            st.write(f"이미지 {img_ok}/{len(sc)} · 음성 {aud_ok}/{len(sc)}")
            ready = sc and img_ok==len(sc) and aud_ok==len(sc)
            if not ready:
                st.warning("모든 장면의 이미지와 음성이 채워져야 영상을 만들 수 있습니다.")
            if st.button("최종 MP4 생성", key="v_build", disabled=not ready):
                with st.spinner("MoviePy로 영상 조립 중... (장면 수에 따라 몇 분 소요)"):
                    try:
                        out=build_video(sel, data)
                        meta["stage"]="04_동영상"; save_project(meta)
                        st.success(f"완성: {out}")
                    except Exception as e:
                        st.error(f"영상 생성 실패: {e}")
            mp4=vdir/"final.mp4"
            if mp4.exists():
                st.video(str(mp4))
                st.caption(f"파일 위치: {mp4}")
