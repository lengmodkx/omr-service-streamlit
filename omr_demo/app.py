"""
答题卡智能处理系统 - Streamlit Demo
支持：模板OMR识别 + 手动标定区域批量裁剪
"""
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas

# 确保能导入core
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.processor import CardProcessor
from core.golden_template import GoldenTemplate
from core.score_calculator import calc_total_score, ScoringConfig
from core.recognizer import make_recognizer, RecognizeContext
from core.recognizer_manager import RecognizerManager

st.set_page_config(page_title="答题卡智能处理系统", layout="wide")

# ========== 初始化 Session State ==========
def init_state():
    defaults = {
        "processor": None,
        "blank_a": None,
        "blank_b": None,
        "results": [],
        "standard_answers": {},
        "manual_corrections": {},
        # 手动标定区域相关（分A/B面）
        "manual_regions_a": [],        # A面标定的区域 [{name, x1, y1, x2, y2}, ...]
        "manual_regions_b": [],        # B面标定的区域
        "ref_image_size_a": None,      # A面标定参考图尺寸 (w, h)
        "ref_image_size_b": None,      # B面标定参考图尺寸 (w, h)
        "crop_results": [],            # 手动裁剪结果
        # 自定义选项框标定（绕过模板bubbles）
        "custom_bubbles": [],           # 用户自定义选项框列表
        "custom_bubbles_img_size": None,  # 标定时图片尺寸 (w, h)
        # 黄金模板标定
        "golden_image": None,           # 黄金模板图片（正确的填涂答题卡）
        "golden_column_boxes": [],      # 用户画的列框 [{x1,y1,x2,y2}, ...]
        "golden_column_configs": [],    # 列框完整参数 [{start_q,num_q,num_options,...}]
        "golden_template": None,        # GoldenTemplate 实例
        "golden_answers": {},           # 黄金模板自动识别的答案 {q: opt}
        "golden_results": [],           # 黄金模板批量识别结果
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # 兼容旧数据：将 manual_regions 迁移到 manual_regions_a，并补充 type 字段
    if "manual_regions" in st.session_state and st.session_state.manual_regions:
        if not st.session_state.manual_regions_a:
            st.session_state.manual_regions_a = st.session_state.manual_regions.copy()
        del st.session_state.manual_regions
    # 给旧数据补充 type 字段（默认为非选择题）
    for regions in [st.session_state.manual_regions_a, st.session_state.manual_regions_b]:
        for r in regions:
            if "type" not in r:
                r["type"] = "非选择题"

init_state()

def on_golden_upload():
    """文件上传回调：将图片存入 session_state，避免 rerun 导致上传文件丢失"""
    f = st.session_state.get("golden_upload")
    if f is not None:
        decoded = cv2.imdecode(np.frombuffer(f.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is not None:
            st.session_state.golden_image = decoded

st.title("📄 答题卡智能处理系统 (Demo版)")
st.caption("流程：上传空白模板 → 标定参考 → 批量处理 → 人工核对 → 导出成绩 | 或：上传参考图 → 手动标定区域 → 批量裁剪")

# ========== 侧边栏 ==========
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # 模板加载
    template_dir = Path(__file__).parent / "templates"
    templates = list(template_dir.glob("*.json"))
    template_names = [t.stem for t in templates]
    
    if template_names:
        sel = st.selectbox("选择答题卡模板", template_names, index=template_names.index("english") if "english" in template_names else 0)
        if st.button("加载模板") or st.session_state.processor is None:
            try:
                st.session_state.processor = CardProcessor(str(template_dir / f"{sel}.json"))
                st.success(f"✅ 已加载模板: {sel}")
            except Exception as e:
                st.error(f"加载失败: {e}")
    else:
        st.warning("templates目录下没有找到模板文件")
    
    st.divider()
    st.header("📋 标准答案设置")
    st.caption("输入格式：每行一题，如 `1:A` 或 `1 A`")
    ans_text = st.text_area("标准答案", 
                            value="\n".join([f"{k}:{v}" for k,v in st.session_state.standard_answers.items()]) if st.session_state.standard_answers else "",
                            placeholder="1:A\n2:B\n3:C\n...", height=120)
    if st.button("导入标准答案"):
        ans = {}
        for line in ans_text.strip().split("\n"):
            line = line.strip().replace(" ", ":").replace("，", ":").replace(",", ":")
            if ":" in line:
                parts = line.split(":")
                try:
                    q = int(parts[0].strip())
                    a = parts[1].strip().upper()
                    ans[q] = a
                except:
                    pass
        st.session_state.standard_answers = ans
        st.success(f"已导入 {len(ans)} 题标准答案")
    
    if st.session_state.standard_answers:
        st.info(f"当前已配置 {len(st.session_state.standard_answers)} 题")

# ========== Tab 导航 ==========
tab_names = ["🖼️ 模板与参考", "📤 批量处理", "📊 结果核对与导出"]
tab1, tab2, tab3 = st.tabs(tab_names)

# ---------- Tab 1: 模板与参考 + 手动标定区域 ----------
with tab1:
    # ===== 手动标定截取区域 =====
    st.header("1. 手动标定截取区域（用于批量裁剪）")
    st.info("上传A面/B面答题卡样本图片，在图片上拖拽画框标定需要截取的区域。")

    c1, c2 = st.columns(2)
    with c1:
        blank_a_file = st.file_uploader("A面参考图片", type=["jpg", "jpeg", "png"], key="blank_a_up")
        if blank_a_file:
            bytes_a = np.asarray(bytearray(blank_a_file.read()), dtype=np.uint8)
            st.session_state.blank_a = cv2.imdecode(bytes_a, cv2.IMREAD_COLOR)
    with c2:
        blank_b_file = st.file_uploader("B面参考图片", type=["jpg", "jpeg", "png"], key="blank_b_up")
        if blank_b_file:
            bytes_b = np.asarray(bytearray(blank_b_file.read()), dtype=np.uint8)
            st.session_state.blank_b = cv2.imdecode(bytes_b, cv2.IMREAD_COLOR)

    has_a = st.session_state.blank_a is not None
    has_b = st.session_state.blank_b is not None

    if not has_a and not has_b:
        st.warning("请先上传A面或B面参考图片")
    else:
        MAX_CANVAS_WIDTH = 500

        def _render_side(col, is_a):
            """渲染单面（A或B）的标定 UI"""
            side_label = "A面" if is_a else "B面"
            ref_img = st.session_state.blank_a if is_a else st.session_state.blank_b
            regions_key = "manual_regions_a" if is_a else "manual_regions_b"
            regions_list = st.session_state[regions_key]
            size_key = "ref_image_size_a" if is_a else "ref_image_size_b"
            kp = "ra" if is_a else "rb"

            with col:
                st.subheader(f"{side_label}")
                if ref_img is None:
                    st.caption("未上传")
                    return

                h, w = ref_img.shape[:2]
                st.session_state[size_key] = (w, h)
                st.caption(f"尺寸: {w}×{h}")

                scale = MAX_CANVAS_WIDTH / w
                dw = MAX_CANVAS_WIDTH
                dh = int(h * scale)

                bg = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
                bg = bg.convert("RGBA")
                bg = bg.resize((dw, dh))

                cv_key = f"canvas_ver_{kp}"
                canvas_ver = st.session_state.get(cv_key, 0)
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.2)",
                    stroke_width=2,
                    stroke_color="#FF0000",
                    background_image=bg,
                    height=dh,
                    width=dw,
                    drawing_mode="rect",
                    key=f"canvas_{kp}_{canvas_ver}",
                    update_streamlit=True,
                )

                col_add, col_clr = st.columns([1, 1])
                with col_add:
                    if st.button("添加画框", key=f"btn_add_{kp}"):
                        if canvas_result.json_data is not None:
                            rects = [obj for obj in canvas_result.json_data.get("objects", [])
                                     if obj.get("type") == "rect"]
                            for obj in rects:
                                regions_list.append({
                                    "name": f"区域{len(regions_list)+1}",
                                    "x1": int(obj["left"] / scale),
                                    "y1": int(obj["top"] / scale),
                                    "x2": int((obj["left"] + obj["width"]) / scale),
                                    "y2": int((obj["top"] + obj["height"]) / scale),
                                    "type": "非选择题",
                                })
                            st.session_state[regions_key] = regions_list
                            st.session_state[cv_key] = canvas_ver + 1
                            st.rerun()
                with col_clr:
                    if st.button("清空画布", key=f"btn_clear_{kp}"):
                        st.session_state[cv_key] = canvas_ver + 1
                        st.rerun()

                # 坐标输入
                with st.expander("或手动输入坐标"):
                    cx1, cy1, cx2, cy2 = st.columns(4)
                    with cx1:
                        mx1 = st.number_input("x1", 0, w, 0, key=f"mx1_{kp}")
                    with cy1:
                        my1 = st.number_input("y1", 0, h, 0, key=f"my1_{kp}")
                    with cx2:
                        mx2 = st.number_input("x2", 0, w, min(w, 200), key=f"mx2_{kp}")
                    with cy2:
                        my2 = st.number_input("y2", 0, h, min(h, 200), key=f"my2_{kp}")
                    if st.button("添加坐标", key=f"btn_manual_{kp}"):
                        regions_list.append({
                            "name": f"区域{len(regions_list)+1}",
                            "x1": int(mx1), "y1": int(my1),
                            "x2": int(mx2), "y2": int(my2),
                            "type": "非选择题",
                        })
                        st.session_state[regions_key] = regions_list
                        st.rerun()

                # 区域列表
                if regions_list:
                    st.caption(f"已添加 {len(regions_list)} 个区域")
                    for i, region in enumerate(regions_list):
                        c = st.columns([2, 1, 1, 1, 1, 0.8])
                        with c[0]:
                            st.text_input("名称", region["name"], key=f"{kp}_n_{i}", label_visibility="collapsed")
                        with c[1]:
                            st.number_input("x1", region["x1"], key=f"{kp}_x1_{i}", label_visibility="collapsed")
                        with c[2]:
                            st.number_input("y1", region["y1"], key=f"{kp}_y1_{i}", label_visibility="collapsed")
                        with c[3]:
                            st.number_input("x2", region["x2"], key=f"{kp}_x2_{i}", label_visibility="collapsed")
                        with c[4]:
                            st.number_input("y2", region["y2"], key=f"{kp}_y2_{i}", label_visibility="collapsed")
                        with c[5]:
                            st.selectbox("类型", ["非选择题", "选择题", "个人信息"],
                                         index=["非选择题", "选择题", "个人信息"].index(region.get("type", "非选择题")),
                                         key=f"{kp}_tp_{i}", label_visibility="collapsed")
                        # 删除按钮（每行独立）
                        if st.button("删", key=f"{kp}_del_{i}"):
                            regions_list.pop(i)
                            st.session_state[regions_key] = regions_list
                            st.rerun()
                else:
                    st.caption("暂无区域")

                # 同步
                synced = []
                for i in range(len(regions_list)):
                    synced.append({
                        "name": st.session_state.get(f"{kp}_n_{i}", regions_list[i]["name"]),
                        "x1": st.session_state.get(f"{kp}_x1_{i}", regions_list[i]["x1"]),
                        "y1": st.session_state.get(f"{kp}_y1_{i}", regions_list[i]["y1"]),
                        "x2": st.session_state.get(f"{kp}_x2_{i}", regions_list[i]["x2"]),
                        "y2": st.session_state.get(f"{kp}_y2_{i}", regions_list[i]["y2"]),
                        "type": st.session_state.get(f"{kp}_tp_{i}", regions_list[i].get("type", "非选择题")),
                    })
                st.session_state[regions_key] = synced

                # 预览
                if synced:
                    with st.expander("查看预览"):
                        vis = ref_img.copy()
                        colors = {"非选择题": (0, 255, 0), "选择题": (255, 0, 0), "个人信息": (0, 0, 255)}
                        for r in synced:
                            color = colors.get(r["type"], (0, 255, 0))
                            cv2.rectangle(vis, (r["x1"], r["y1"]), (r["x2"], r["y2"]), color, 2)
                            cv2.putText(vis, f"{r['name']}({r['type'][:2]})",
                                        (r["x1"], r["y1"] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_column_width=True)

        col_a, col_b = st.columns(2)
        _render_side(col_a, True)
        _render_side(col_b, False)

        # 导出/导入
        st.divider()
        col_exp, col_imp = st.columns(2)
        with col_exp:
            export_data = {
                "image_size_a": {"w": st.session_state.ref_image_size_a[0], "h": st.session_state.ref_image_size_a[1]} if st.session_state.ref_image_size_a else None,
                "image_size_b": {"w": st.session_state.ref_image_size_b[0], "h": st.session_state.ref_image_size_b[1]} if st.session_state.ref_image_size_b else None,
                "regions_a": st.session_state.manual_regions_a,
                "regions_b": st.session_state.manual_regions_b,
            }
            st.download_button("导出配置(JSON)", json.dumps(export_data, ensure_ascii=False, indent=2),
                               file_name="manual_regions.json", mime="application/json")
        with col_imp:
            imported = st.file_uploader("导入配置", type=["json"], key="import_regions")
            if imported:
                try:
                    data = json.loads(imported.read().decode("utf-8"))
                    if "regions_a" in data:
                        st.session_state.manual_regions_a = data["regions_a"]
                    if "regions_b" in data:
                        st.session_state.manual_regions_b = data["regions_b"]
                    if "regions" in data and "regions_a" not in data:
                        st.session_state.manual_regions_a = data["regions"]
                    st.success("已导入")
                    st.rerun()
                except Exception as e:
                    st.error(f"导入失败: {e}")
    
    # ===== 黄金模板标定 =====
    st.divider()
    st.header("2. 黄金模板标定")
    st.info("上传一份**正确填涂**的答题卡，画出选择题列区域，系统自动识别答案并保存为黄金模板。后续批量处理时用此模板比对识别。")

    golden_img_file = st.file_uploader(
        "上传正确填涂的答题卡", type=["jpg", "jpeg", "png"],
        key="golden_upload", on_change=on_golden_upload)

    if st.session_state.golden_image is not None:
        gimg = st.session_state.golden_image
        gh, gw = gimg.shape[:2]
        st.caption(f"图片尺寸: {gw} × {gh}  |  下方画布叠加了该图片作为背景，可直接拖拽画框")

        # Canvas 画列框
        MAX_CANVAS_WIDTH = 750
        g_canvas_scale = MAX_CANVAS_WIDTH / gw
        g_display_w = MAX_CANVAS_WIDTH
        g_display_h = int(gh * g_canvas_scale)

        g_bg = Image.fromarray(cv2.cvtColor(gimg, cv2.COLOR_BGR2RGB))
        g_bg = g_bg.convert("RGBA")
        g_bg = g_bg.resize((g_display_w, g_display_h))

        st.markdown("**画列框**：在图片上拖拽画出每列选择题的矩形区域")
        st.caption("从左到右依次画框，每个框围住一列选择题的所有气泡")

        g_canvas_ver = st.session_state.get("golden_canvas_ver", 0)
        g_canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=g_bg,
            height=g_display_h,
            width=g_display_w,
            drawing_mode="rect",
            key=f"golden_canvas_{g_canvas_ver}",
            update_streamlit=True,
        )

        col_ext, col_clr = st.columns([1, 1])
        with col_ext:
            if st.button("提取画框", key="golden_extract"):
                json_data = g_canvas_result.json_data
                if json_data is None:
                    st.warning("未获取到画布数据，请确保已在画布上画了矩形框再点击提取")
                else:
                    rects = [obj for obj in json_data.get("objects", [])
                             if obj.get("type") == "rect"]
                    if not rects:
                        st.warning("未检测到矩形框，请在画布上拖拽画出列框后再点击提取")
                    else:
                        boxes = []
                        for obj in rects:
                            boxes.append({
                                "x1": int(obj["left"] / g_canvas_scale),
                                "y1": int(obj["top"] / g_canvas_scale),
                                "x2": int((obj["left"] + obj["width"]) / g_canvas_scale),
                                "y2": int((obj["top"] + obj["height"]) / g_canvas_scale),
                            })
                        st.session_state.golden_column_boxes = boxes
                        st.session_state.golden_canvas_ver = g_canvas_ver + 1
                        st.success(f"已提取 {len(boxes)} 个列框")
                        st.rerun()
        with col_clr:
            if st.button("清空画布", key="golden_clear"):
                st.session_state.golden_canvas_ver = g_canvas_ver + 1
                st.session_state.golden_column_boxes = []
                st.session_state.golden_column_configs = []
                st.rerun()

        # 列框参数配置
        if st.session_state.golden_column_boxes:
            boxes = st.session_state.golden_column_boxes
            st.markdown(f"**已提取 {len(boxes)} 个列框，配置每题参数：**")

            # 批量默认值
            bc1, bc2, bc3 = st.columns([1, 1, 1])
            with bc1:
                batch_nq = st.number_input("默认题目数", 1, 20, 5, key="golden_batch_nq")
            with bc2:
                batch_no = st.number_input("默认选项数", 2, 7, 4, key="golden_batch_no")
            with bc3:
                if st.button("应用到全部列", key="golden_batch_apply"):
                    for i in range(len(boxes)):
                        st.session_state[f"gc_nq_{i}"] = int(batch_nq)
                        st.session_state[f"gc_no_{i}"] = int(batch_no)
                    st.rerun()

            configs = []
            for i, box in enumerate(boxes):
                cols = st.columns([1, 1, 1, 1])
                with cols[0]:
                    sq = st.number_input("起始题号", min_value=1, value=i * 5 + 1, key=f"gc_sq_{i}")
                with cols[1]:
                    nq = st.number_input("题目数", min_value=1,
                                         value=st.session_state.get(f"gc_nq_{i}", 5),
                                         key=f"gc_nq_{i}")
                with cols[2]:
                    no = st.number_input("选项数", min_value=2,
                                         value=st.session_state.get(f"gc_no_{i}", 4),
                                         key=f"gc_no_{i}")
                with cols[3]:
                    st.caption(f"框: ({box['x1']},{box['y1']})-({box['x2']},{box['y2']})")
                configs.append({
                    "x1": box["x1"], "y1": box["y1"],
                    "x2": box["x2"], "y2": box["y2"],
                    "start_q": int(sq), "num_q": int(nq), "num_options": int(no),
                })
            st.session_state.golden_column_configs = configs

            # 生成并保存黄金模板
            if st.button("生成黄金模板并自动识别答案", type="primary"):
                gtp = GoldenTemplate(gimg, configs)
                st.session_state.golden_template = gtp
                st.session_state.golden_answers = gtp.answers
                st.success(f"黄金模板已生成！{len(gtp.bubbles)} 个气泡，{len(gtp.answers)} 题答案")
                st.rerun()

    # 核对与修正答案
    if st.session_state.golden_template is not None:
        gtp = st.session_state.golden_template
        st.divider()
        st.subheader("核对黄金模板答案")
        st.caption("检查自动识别的答案是否正确，如有误请修正")

        ans_data = []
        for q in sorted(gtp.answers.keys()):
            ans_data.append({
                "题号": q,
                "识别答案": gtp.answers.get(q) or "(未识别)",
            })
        if ans_data:
            df_ans = pd.DataFrame(ans_data)
            edited = st.data_editor(
                df_ans,
                column_config={
                    "识别答案": st.column_config.TextColumn("识别答案", help="修改为标准答案"),
                },
                hide_index=True,
                height=400,
                key="golden_answer_editor",
            )

            with st.columns([1, 1, 3])[0]:
                if st.button("保存修正"):
                    for _, row in edited.iterrows():
                        val = str(row["识别答案"]).strip().upper()
                        if val and val != "(未识别)" and val != "NONE":
                            gtp.calibrate_answer(int(row["题号"]), val)
                    st.session_state.golden_answers = gtp.answers
                    st.success(f"已保存 {len(gtp.answers)} 题标准答案")
                    st.rerun()

            # 显示未识别题目的调试信息
            if hasattr(gtp, '_debug_samples') and gtp._debug_samples:
                unrecognized = [d for d in gtp._debug_samples if d["answer"] is None]
                if unrecognized:
                    with st.expander(f"调试信息：{len(unrecognized)} 题未识别（查看采样值）", expanded=True):
                        lines = []
                        for d in unrecognized:
                            opts_str = " | ".join(f"{o}={v}" for o, v in d["opts"].items())
                            lines.append(
                                f"Q{d['q']:>2}: best={d['best_opt']}={d['best_val']} "
                                f"gap={d['gap']} mean={d['mean_val']} | {opts_str}"
                            )
                        st.code("\n".join(lines), language=None)
                else:
                    with st.expander("调试信息：全部识别成功", expanded=False):
                        lines = []
                        for d in gtp._debug_samples[:5]:
                            opts_str = " | ".join(f"{o}={v}" for o, v in d["opts"].items())
                            lines.append(
                                f"Q{d['q']:>2}: ans={d['answer']} best={d['best_opt']}={d['best_val']} "
                                f"gap={d['gap']} | {opts_str}"
                            )
                        st.code("\n".join(lines), language=None)
                        if len(gtp._debug_samples) > 5:
                            st.caption(f"... 共 {len(gtp._debug_samples)} 题")

        # 预览气泡覆盖
        with st.expander("查看气泡覆盖预览"):
            vis = gimg.copy()
            # 画列框边界和内部网格线
            for cfg in st.session_state.golden_column_configs:
                x1, y1, x2, y2 = cfg["x1"], cfg["y1"], cfg["x2"], cfg["y2"]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                row_h = (y2 - y1) / cfg["num_q"]
                for qi in range(1, cfg["num_q"]):
                    ly = int(y1 + qi * row_h)
                    cv2.line(vis, (x1, ly), (x2, ly), (0, 255, 255), 1)
            # 画气泡采样点
            for b in gtp.bubbles:
                cv2.circle(vis, (b["x"], b["y"]), max(4, b["w"] // 2), (0, 255, 0), 2)
                cv2.line(vis, (b["x"]-3, b["y"]), (b["x"]+3, b["y"]), (0, 0, 255), 2)
                cv2.line(vis, (b["x"], b["y"]-3), (b["x"], b["y"]+3), (0, 0, 255), 2)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                     caption="黄色=列框与分隔线 | 绿色=采样窗口 | 红色十字=采样中心", use_column_width=True)

# ---------- Tab 2: 批量处理 ----------
with tab2:
    st.header("批量处理答题卡")
    st.caption("使用标定的截取区域裁剪图片，并用黄金模板识别选择题答案")

    has_a = bool(st.session_state.manual_regions_a)
    has_b = bool(st.session_state.manual_regions_b)
    has_golden = st.session_state.golden_template is not None

    if not has_a and not has_b:
        st.warning("请先在「模板与参考」页面标定截取区域")
    elif not has_golden:
        st.warning("请先在「模板与参考」页面第三步「黄金模板标定」中生成黄金模板")
    else:
        col_status = st.columns(3)
        with col_status[0]:
            st.success(f"A面 {len(st.session_state.manual_regions_a)} 个区域")
        with col_status[1]:
            st.success(f"B面 {len(st.session_state.manual_regions_b)} 个区域")
        with col_status[2]:
            st.success(f"黄金模板 {len(st.session_state.golden_answers)} 题答案")

        uploaded = st.file_uploader("上传答题卡图片（A+B配对，文件名需对应如 `xxx01A.jpg` / `xxx01B.jpg`）",
                                    type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch_unified")

        if uploaded:
            pairs = {}
            for f in uploaded:
                name = f.name.upper()
                base = None
                side = None
                if name.endswith("A.JPG") or name.endswith("A.JPEG") or name.endswith("A.PNG"):
                    base = f.name[:-5] if name.endswith("A.JPG") else (f.name[:-6] if name.endswith("A.JPEG") else f.name[:-5])
                    side = "A"
                elif name.endswith("B.JPG") or name.endswith("B.JPEG") or name.endswith("B.PNG"):
                    base = f.name[:-5] if name.endswith("B.JPG") else (f.name[:-6] if name.endswith("B.JPEG") else f.name[:-5])
                    side = "B"
                if base and side:
                    pairs.setdefault(base, {})[side] = f

            valid = {k: v for k, v in pairs.items() if "A" in v and "B" in v}
            single_a = {k: v for k, v in pairs.items() if "A" in v and "B" not in v}
            single_b = {k: v for k, v in pairs.items() if "B" in v and "A" not in v}
            st.write(f"识别到 **{len(valid)}** 组A+B配对，**{len(single_a)}** 张单A面，**{len(single_b)}** 张单B面")

            debug_mode = st.checkbox("调试模式（输出每题详细采样值）", key="debug_unified")
            enable_cv = st.checkbox(
                "启用双识别器交叉验证 (黄金模板 + 差分法, 慢但更准)",
                value=False, key="enable_cross_validate",
                help="同一张卡跑两个识别器,分歧题自动进人工核对面板"
            )

            if st.button("开始处理", type="primary"):
                gtp = st.session_state.golden_template
                output_dir = os.path.join("output", "batch")
                results = []
                crop_summary = []

                bar = st.progress(0)
                status = st.empty()
                total = len(valid) + len(single_a) + len(single_b)
                processed = [0]  # 用列表包装，避免 nonlocal 问题

                def _process_single(key, file_a, file_b):
                    processed[0] += 1
                    status.info(f"处理中 [{processed[0]}/{total}]: {key}")

                    fa = file_a; fa.seek(0)
                    img_a = cv2.imdecode(np.frombuffer(fa.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img_a is None:
                        return None

                    # 1. 黄金模板识别选择题（A面）— 走 Recognizer 协议入口
                    if enable_cv:
                        # 双识别器交叉验证 (黄金模板 + 差分法)
                        golden_rec = make_recognizer("golden", golden_template=gtp)
                        diff_rec = make_recognizer(
                            "differential",
                            processor=st.session_state.processor,
                            page="A",
                        )
                        manager = RecognizerManager([golden_rec, diff_rec])
                        result = manager.cross_validate(
                            img_a,
                            RecognizeContext(standard_answers=st.session_state.standard_answers),
                        )
                    else:
                        # 单识别器(原有路径)
                        recognizer = make_recognizer("golden", golden_template=gtp)
                        result = recognizer.recognize(
                            img_a,
                            RecognizeContext(standard_answers=st.session_state.standard_answers),
                        )
                    r = result.to_legacy_dict()        # 转回 dict 形态,下游代码 0 改动
                    r["_key"] = key
                    r["_file_a"] = file_a.name
                    r["_file_b"] = file_b.name if file_b else ""
                    # 交叉验证结果字段(未启用时为占位空值,Tab3 展示 0 差异)
                    r["_disputed_questions"] = getattr(result, "disputed_questions", [])
                    r["_agreement_rate"] = getattr(result, "agreement_rate", 1.0)

                    # 1b. 生成识别预览图（气泡采样点叠加在原图上）
                    os.makedirs(output_dir, exist_ok=True)
                    preview = img_a.copy()
                    h_p, w_p = preview.shape[:2]
                    gt_img = gtp.image
                    gt_h, gt_w = gt_img.shape[:2]
                    for b in gtp.bubbles:
                        q, opt = b["q"], b["opt"]
                        # 坐标从黄金模板尺寸缩放到实际图像尺寸
                        sx = int(b["x"] * w_p / gt_w)
                        sy = int(b["y"] * h_p / gt_h)
                        ans_info = r["answers"].get(q, {})
                        sts = ans_info.get("status", "uncertain")
                        detected = ans_info.get("answer", "")
                        is_detected = (opt in detected) if detected else False
                        if is_detected and sts == "multi":
                            color = (255, 0, 0)  # 蓝色
                        elif is_detected:
                            color = (0, 200, 0)  # 绿色
                        else:
                            color = (180, 180, 180)  # 灰色
                        cv2.circle(preview, (sx, sy), 6, color, 2 if is_detected else 1)
                    preview_path = os.path.join(output_dir, f"{key}_golden_preview.png")
                    _, png_buf = cv2.imencode(".png", preview)
                    with open(preview_path, "wb") as pf:
                        pf.write(png_buf)
                    r["_preview_path"] = preview_path

                    # 2. 计分（与黄金答案对比）
                    correct = 0
                    total_q = r["total"]
                    for ans in r["answers"].values():
                        if ans.get("correct") is True:
                            correct += 1
                    r["_score"] = correct
                    r["_total"] = total_q

                    # 白卷检测：识别率低于50%
                    total_q = r["total"]
                    answered = total_q - r["empty_count"]
                    rate = answered / total_q * 100 if total_q > 0 else 0
                    r["_is_blank"] = rate < 50
                    if r["_is_blank"]:
                        r["_score"] = 0

                    # 3. 截取区域裁剪（A面）
                    if has_a:
                        crops_a = CardProcessor.crop_by_regions(
                            img_a, st.session_state.manual_regions_a,
                            output_dir, f"{key}_A",
                            ref_size=st.session_state.ref_image_size_a)
                        for c in crops_a:
                            crop_summary.append({"key": key, "文件": file_a.name, "面别": "A", "区域": c["name"], "路径": c["path"]})

                    # 4. 截取区域裁剪（B面）
                    if file_b is not None and has_b:
                        fb = file_b; fb.seek(0)
                        img_b = cv2.imdecode(np.frombuffer(fb.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img_b is not None:
                            crops_b = CardProcessor.crop_by_regions(
                                img_b, st.session_state.manual_regions_b,
                                output_dir, f"{key}_B",
                                ref_size=st.session_state.ref_image_size_b)
                            for c in crops_b:
                                crop_summary.append({"key": key, "文件": file_b.name, "面别": "B", "区域": c["name"], "路径": c["path"]})

                    return r

                # 处理A+B配对
                for key, files in valid.items():
                    r = _process_single(key, files["A"], files["B"])
                    if r:
                        results.append(r)
                    bar.progress(int(processed[0] / total * 100))

                # 处理单A面
                for key, files in single_a.items():
                    r = _process_single(key, files["A"], None)
                    if r:
                        results.append(r)
                    bar.progress(int(processed[0] / total * 100))

                # 处理单B面（仅裁剪，无选择题识别）
                for key, files in single_b.items():
                    processed[0] += 1
                    status.info(f"处理中 [{processed[0]}/{total}]: {key} (仅B面裁剪)")
                    fb = files["B"]; fb.seek(0)
                    img_b = cv2.imdecode(np.frombuffer(fb.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img_b is not None and has_b:
                        crops_b = CardProcessor.crop_by_regions(
                            img_b, st.session_state.manual_regions_b,
                            output_dir, f"{key}_B",
                            ref_size=st.session_state.ref_image_size_b)
                        for c in crops_b:
                            crop_summary.append({"key": key, "文件": files["B"].name, "面别": "B", "区域": c["name"], "路径": c["path"]})
                    bar.progress(int(processed[0] / total * 100))

                status.empty()
                bar.empty()
                st.session_state.results = results
                st.session_state.crop_results = crop_summary
                total_crops = len(crop_summary)
                st.success(f"处理完成！共 {len(results)} 份识别结果，{total_crops} 个裁剪区域")

                if total_crops == 0 and (has_a or has_b):
                    st.warning("未生成任何裁剪区域，请检查「模板与参考」页面的截取区域是否正确配置")

                # 摘要
                if results:
                    rows = []
                    for r in results:
                        total_q = r["total"]
                        answered = total_q - r["empty_count"]
                        rate = answered / total_q * 100 if total_q > 0 else 0
                        is_blank = rate < 50
                        rows.append({
                            "学生/文件": r["_key"],
                            "识别题数": f"{answered}/{total_q}",
                            "识别率": f"{rate:.0f}%",
                            "状态": "白卷" if is_blank else "正常",
                            "得分": f"{r.get('_score', 0)}/{r.get('_total', 0)}",
                            "漏涂": r["empty_count"],
                            "多选": r["multi_count"],
                            "分歧": len(r.get("_disputed_questions", [])),
                            "一致率": f"{r.get('_agreement_rate', 1.0)*100:.0f}%",
                        })
                    df = pd.DataFrame(rows)
                    st.dataframe(df)

                    blank_count = sum(1 for row in rows if row["状态"] == "白卷")
                    if blank_count > 0:
                        st.warning(f"检测到 {blank_count} 份白卷（识别率低于50%），已自动标记")

                if crop_summary:
                    with st.expander(f"裁剪详情（{len(crop_summary)} 个区域）"):
                        st.dataframe(pd.DataFrame(crop_summary))

# ---------- Tab 3: 结果核对与导出 ----------
with tab3:
    st.header("结果核对与导出")

    if not st.session_state.results and not st.session_state.crop_results:
        st.info("请先在「批量处理」页面上传并处理答题卡")
    else:
        # ===== 选择题识别结果 =====
        if st.session_state.results:
            results = st.session_state.results
            options = [f"{r['_key']} ({r['_file_a']})" for r in results]
            sel_label = st.selectbox("选择答题卡查看详情", options)
            sel_idx = options.index(sel_label)
            result = results[sel_idx]
            sid = result["_key"]
            blank_tag = " [白卷]" if result.get("_is_blank") else ""
            st.subheader(f"📋 {sid}{blank_tag} - 识别详情")
            if result.get("_is_blank"):
                st.warning("该试卷识别率低于50%，判定为白卷，得分已置为0")

            # 交叉验证分歧题提示(阶段 7 新增)
            disputed = result.get("_disputed_questions", [])
            agreement = result.get("_agreement_rate", 1.0)
            if disputed:
                preview_list = disputed[:8]
                more = f" 等共 {len(disputed)} 题" if len(disputed) > 8 else ""
                st.warning(
                    f"⚠️ 双识别器交叉验证发现 {len(disputed)} 个分歧题: "
                    f"{preview_list}{more}，识别器一致率 {agreement*100:.0f}%。"
                    f"建议优先人工核对分歧题。"
                )

            # 黄金模板识别预览图
            preview_path = result.get("_preview_path")
            if preview_path and os.path.exists(preview_path):
                with st.expander("🔍 黄金模板识别预览（绿色=已识别选项，灰色=未识别，蓝色=多选）", expanded=False):
                    st.image(preview_path, use_column_width=True)

            # 调试信息
            debug_lines = result.get("debug_lines", [])
            if debug_lines:
                with st.expander(f"调试信息：{len(debug_lines)} 题未识别", expanded=True):
                    st.code("\n".join(debug_lines), language=None)

            st.markdown("**选择题识别结果（可人工修正）**")
            corrections = st.session_state.manual_corrections.get(sid, {})
            all_qs = sorted(result["answers"].keys())

            choice_data = []
            for q in all_qs:
                ans_info = result["answers"].get(q, {})
                auto_ans = ans_info.get("answer", "")
                auto_display = auto_ans if auto_ans else "(未识别)"
                if ans_info.get("status") == "multi":
                    auto_display += "(多选)"
                std = st.session_state.golden_answers.get(q, "")
                corrected = corrections.get(q)

                corr = ans_info.get("correct")
                if corr is True:
                    status_str = "✅"
                elif corr is False:
                    status_str = "❌"
                elif not auto_ans:
                    status_str = "⚪"
                else:
                    status_str = ""

                choice_data.append({
                    "题号": q,
                    "自动识别": auto_display,
                    "人工修正": corrected if corrected else "",
                    "标准答案": std,
                    "状态": status_str,
                    "_q": q
                })

            edited = st.data_editor(
                pd.DataFrame(choice_data)[["题号", "自动识别", "人工修正", "标准答案", "状态"]],
                column_config={
                    "人工修正": st.column_config.TextColumn("人工修正", help="如需修正，直接输入A/B/C/D等"),
                },
                hide_index=True,
                height=500,
                key="golden_result_editor",
            )

            new_corr = {}
            for _, row in edited.iterrows():
                if row["人工修正"] and row["人工修正"].strip():
                    new_corr[int(row["题号"])] = row["人工修正"].strip().upper()
            st.session_state.manual_corrections[sid] = new_corr

            # 基于人工修正重新计分
            golden_ans = st.session_state.golden_answers
            # 构造 effective_answers: 人工修正优先,否则用识别器结果
            effective_answers = {}
            for q, std_ans in golden_ans.items():
                corrected = new_corr.get(q)
                if corrected:
                    effective_answers[q] = {"answer": corrected, "status": "single"}
                else:
                    ans_info = result["answers"].get(q, {})
                    effective_answers[q] = {
                        "answer": ans_info.get("answer"),
                        "status": ans_info.get("status", "empty"),
                    }
            score_result = calc_total_score(effective_answers, golden_ans, ScoringConfig())
            sc = score_result["total"]
            tot = score_result["total_full"]
            st.metric("选择题最终得分", f"{sc} / {tot}")

            # 全局导出
            st.divider()
            st.subheader("📥 批量导出")
            if st.button("生成Excel成绩单"):
                export_rows = []
                for r in results:
                    sid_r = r.get("_key", "unknown")
                    corr_r = st.session_state.manual_corrections.get(sid_r, {})
                    row = {"学生ID": sid_r, "条形码": r.get("barcode", ""),
                           "选择题得分": r.get("_score", 0), "选择题满分": r.get("_total", 0),
                           "白卷": "是" if r.get("_is_blank") else "否",
                           "异常标记": r.get("card_flag") or "",
                           "_answers_json": json.dumps(r.get("answers", {}), ensure_ascii=False)}
                    for q in sorted(r["answers"].keys()):
                        ans = corr_r.get(q)
                        if not ans:
                            ans_info = r["answers"].get(q, {})
                            ans = ans_info.get("answer", "") or ""
                        row[f"Q{q}"] = ans
                    export_rows.append(row)

                df_export = pd.DataFrame(export_rows)
                buf = io.BytesIO()
                df_export.to_excel(buf, index=False, engine="openpyxl")
                buf.seek(0)
                st.download_button(
                    label="⬇️ 下载Excel成绩单",
                    data=buf,
                    file_name="答题卡成绩汇总.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                st.dataframe(df_export.head(20))

        # ===== 裁剪结果 =====
        if st.session_state.crop_results:
            st.divider()
            st.subheader("📁 截取区域裁剪结果")
            cr = st.session_state.crop_results
            if isinstance(cr, list) and cr:
                if isinstance(cr[0], dict) and "文件" in cr[0]:
                    # 新格式（统一处理产生）：按学生分组展示裁剪图片
                    from collections import defaultdict
                    groups = defaultdict(lambda: defaultdict(list))
                    for row in cr:
                        groups[row.get("key", "unknown")][row["面别"]].append(row)
                    sel_key = st.selectbox("选择答题卡查看裁剪图片", sorted(groups.keys()))
                    for side in ["A", "B"]:
                        side_items = groups[sel_key].get(side, [])
                        if side_items:
                            st.markdown(f"**{side}面**")
                            cols = st.columns(3)
                            for idx, item in enumerate(side_items):
                                with cols[idx % 3]:
                                    if Path(item["路径"]).exists():
                                        st.image(Image.open(item["路径"]),
                                                 caption=f"{item['区域']} ({item['面别']}面)",
                                                 use_column_width=True)
                                    else:
                                        st.warning(f"{item['区域']}: 文件未找到")
                elif isinstance(cr[0], dict) and "crops" in cr[0]:
                    # 旧格式兼容
                    from collections import defaultdict
                    groups = defaultdict(list)
                    for item in cr:
                        groups[item.get("key", item["file"])].append(item)
                    sel_key = st.selectbox("选择答题卡查看裁剪图片", sorted(groups.keys()))
                    for item in groups[sel_key]:
                        st.markdown(f"**{item['side']}面 - {item['file']}**")
                        cols = st.columns(3)
                        for idx, crop in enumerate(item["crops"]):
                            with cols[idx % 3]:
                                if Path(crop["path"]).exists():
                                    st.image(Image.open(crop["path"]), caption=crop["name"], use_column_width=True)
                                else:
                                    st.warning(f"{crop['name']}: 文件未找到")
