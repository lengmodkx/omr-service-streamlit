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
from core.standard_template import StandardTemplate
from core.score_calculator import calc_total_score, ScoringConfig
from core.recognizer import make_recognizer, RecognizeContext

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
        # 选择题所在面: "A" / "B" — 用户在 Tab1 指定,Tab2 据此识别
        "mc_side": "A",
        # 标准模板标定
        "template_image": None,           # 标准模板图片（正确的填涂答题卡）
        "template_column_boxes": [],      # 用户画的列框 [{x1,y1,x2,y2}, ...]
        "template_column_configs": [],    # 列框完整参数 [{start_q,num_q,num_options,...}]
        "standard_template": None,        # StandardTemplate 实例
        "template_answers": {},           # 标准模板自动识别的答案 {q: opt}
        "template_results": [],           # 标准模板批量识别结果
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

def on_template_upload():
    """文件上传回调：将图片存入 session_state，避免 rerun 导致上传文件丢失"""
    f = st.session_state.get("template_upload")
    if f is not None:
        decoded = cv2.imdecode(np.frombuffer(f.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is not None:
            st.session_state.template_image = decoded

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
        # 始终全宽(900px):AB 上下堆叠,各自画布都拿到 900px 画框更精准
        MAX_CANVAS_WIDTH = 900

        def _render_side(col, is_a, max_w=MAX_CANVAS_WIDTH):
            """渲染单面（A或B）的标定 UI。
            max_w: 该面画布的最大宽度(像素),由调用方根据 AB 上传情况决定。"""
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

                scale = max_w / w
                dw = max_w
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

        # 上下堆叠布局:AB 都有时也上下排,各占 900px 全宽,画框更精准
        if has_a:
            _render_side(st.container(), True)
        if has_b:
            if has_a:
                st.divider()  # AB 之间视觉分隔
            _render_side(st.container(), False)

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
    
    # ===== 标准模板标定 =====
    st.divider()
    st.header("2. 标准模板标定")
    st.info("上传一份**正确填涂**的答题卡，画出选择题列区域，系统自动识别答案并保存为标准模板。后续批量处理时用此模板比对识别。")

    # 选择题所在面 — 决定 Tab2 识别哪张图
    # 默认 A 面是常规情况(选择题在 A),遇到"B 面才是含选择题的卷"切到 B
    st.session_state.mc_side = st.radio(
        "选择题所在面",
        options=["A", "B"],
        index=0 if st.session_state.mc_side == "A" else 1,
        key="mc_side_radio",
        horizontal=True,
        help="常规答题卡选择题在 A 面;若 B 面才是含选择题的主卷(如化学单面卷),切到 B",
    )

    template_img_file = st.file_uploader(
        "上传正确填涂的答题卡", type=["jpg", "jpeg", "png"],
        key="template_upload", on_change=on_template_upload)

    if st.session_state.template_image is not None:
        timg = st.session_state.template_image
        th, tw = timg.shape[:2]
        st.caption(f"图片尺寸: {tw} × {th}  |  下方画布叠加了该图片作为背景，可直接拖拽画框")

        # Canvas 画列框 (1500px 适配单面大尺寸答题卡,画框更精准)
        MAX_CANVAS_WIDTH = 1500
        t_canvas_scale = MAX_CANVAS_WIDTH / tw
        t_display_w = MAX_CANVAS_WIDTH
        t_display_h = int(th * t_canvas_scale)

        t_bg = Image.fromarray(cv2.cvtColor(timg, cv2.COLOR_BGR2RGB))
        t_bg = t_bg.convert("RGBA")
        t_bg = t_bg.resize((t_display_w, t_display_h))

        st.markdown("**画列框**：在图片上拖拽画出每列选择题的矩形区域")
        st.caption("从左到右依次画框，每个框围住一列选择题的所有气泡")

        t_canvas_ver = st.session_state.get("template_canvas_ver", 0)

        # 把已存列框(原图坐标)按当前画布 scale 反向缩放,回显到画布
        # 这样:换画布宽度后之前画的框依然在原位,不需要从头画
        existing_boxes = st.session_state.get("template_column_boxes", [])
        if existing_boxes and t_canvas_scale > 0:
            initial_drawing = {
                "version": "4.4.0",
                "objects": [
                    {
                        "type": "rect",
                        "left": box["x1"] * t_canvas_scale,
                        "top": box["y1"] * t_canvas_scale,
                        "width": (box["x2"] - box["x1"]) * t_canvas_scale,
                        "height": (box["y2"] - box["y1"]) * t_canvas_scale,
                        "fill": "rgba(255, 165, 0, 0.2)",
                        "stroke": "#FF0000",
                        "strokeWidth": 2,
                    }
                    for box in existing_boxes
                ],
            }
        else:
            initial_drawing = None

        t_canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=t_bg,
            height=t_display_h,
            width=t_display_w,
            drawing_mode="rect",
            initial_drawing=initial_drawing,
            key=f"template_canvas_{t_canvas_ver}",
            update_streamlit=True,
        )

        col_ext, col_clr = st.columns([1, 1])
        with col_ext:
            if st.button("提取画框", key="template_extract"):
                json_data = t_canvas_result.json_data
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
                                "x1": int(obj["left"] / t_canvas_scale),
                                "y1": int(obj["top"] / t_canvas_scale),
                                "x2": int((obj["left"] + obj["width"]) / t_canvas_scale),
                                "y2": int((obj["top"] + obj["height"]) / t_canvas_scale),
                            })
                        st.session_state.template_column_boxes = boxes
                        st.session_state.template_canvas_ver = t_canvas_ver + 1
                        st.success(f"已提取 {len(boxes)} 个列框")
                        st.rerun()
        with col_clr:
            if st.button("清空画布", key="template_clear"):
                st.session_state.template_canvas_ver = t_canvas_ver + 1
                st.session_state.template_column_boxes = []
                st.session_state.template_column_configs = []
                st.rerun()

        # 列框参数配置
        if st.session_state.template_column_boxes:
            boxes = st.session_state.template_column_boxes
            st.markdown(f"**已提取 {len(boxes)} 个列框，配置每题参数：**")

            # 批量默认值
            bc1, bc2, bc3 = st.columns([1, 1, 1])
            with bc1:
                batch_nq = st.number_input("默认题目数", 1, 20, 5, key="template_batch_nq")
            with bc2:
                batch_no = st.number_input("默认选项数", 2, 7, 4, key="template_batch_no")
            with bc3:
                if st.button("应用到全部列", key="template_batch_apply"):
                    for i in range(len(boxes)):
                        st.session_state[f"tc_nq_{i}"] = int(batch_nq)
                        st.session_state[f"tc_no_{i}"] = int(batch_no)
                    st.rerun()

            configs = []
            for i, box in enumerate(boxes):
                cols = st.columns([1, 1, 1, 1, 1.1, 1.5])
                with cols[0]:
                    sq = st.number_input("起始题号", min_value=1, value=i * 5 + 1, key=f"tc_sq_{i}")
                with cols[1]:
                    nq = st.number_input("题目数", min_value=1,
                                         value=st.session_state.get(f"tc_nq_{i}", 5),
                                         key=f"tc_nq_{i}")
                with cols[2]:
                    no = st.number_input("选项数", min_value=2,
                                         value=st.session_state.get(f"tc_no_{i}", 4),
                                         key=f"tc_no_{i}")
                with cols[3]:
                    # 2026-06-04 新增: 倒序题号选项(用于 OMR0002 蒙文答题卡等倒序排列模板)
                    rv = st.checkbox("倒序", value=False, key=f"tc_rv_{i}",
                                     help="勾选时 Q1 放在题号轴末端(x轴→x2端,y轴→y2端)。配合'选项纵向'使用,适配蒙文答题卡")
                with cols[4]:
                    # 2026-06-08 新增: 选项轴方向
                    # x (默认): ABCD 横排,题号竖排 (标准模板)
                    # y:        ABCD 竖排,题号横排 (OMR0002 蒙文答题卡)
                    oa = st.selectbox(
                        "选项轴",
                        options=["x", "y"],
                        index=0,
                        key=f"tc_oa_{i}",
                        help="x: ABCD 横向、题号纵向(标准);y: ABCD 纵向、题号横向(OMR0002 蒙文答题卡)",
                    )
                with cols[5]:
                    st.caption(f"框: ({box['x1']},{box['y1']})-({box['x2']},{box['y2']})")
                configs.append({
                    "x1": box["x1"], "y1": box["y1"],
                    "x2": box["x2"], "y2": box["y2"],
                    "start_q": int(sq), "num_q": int(nq), "num_options": int(no),
                    "reverse_q": bool(rv),  # 2026-06-04 新增
                    "option_axis": oa,  # 2026-06-08 新增
                })
            st.session_state.template_column_configs = configs

            # 生成并保存标准模板
            if st.button("生成标准模板并自动识别答案", type="primary"):
                std_tpl =StandardTemplate(timg, configs)
                st.session_state.standard_template = std_tpl
                st.session_state.template_answers = std_tpl.answers
                st.success(f"标准模板已生成！{len(std_tpl.bubbles)} 个气泡，{len(std_tpl.answers)} 题答案")
                st.rerun()

    # 核对与修正答案
    if st.session_state.standard_template is not None:
        std_tpl =st.session_state.standard_template
        st.divider()
        st.subheader("核对标准模板答案")
        st.caption("检查自动识别的答案是否正确，如有误请修正")

        ans_data = []
        for q in sorted(std_tpl.answers.keys()):
            ans_data.append({
                "题号": q,
                "识别答案": std_tpl.answers.get(q) or "(未识别)",
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
                key="template_answer_editor",
            )

            with st.columns([1, 1, 3])[0]:
                if st.button("保存修正"):
                    for _, row in edited.iterrows():
                        val = str(row["识别答案"]).strip().upper()
                        if val and val != "(未识别)" and val != "NONE":
                            std_tpl.calibrate_answer(int(row["题号"]), val)
                    st.session_state.template_answers = std_tpl.answers
                    st.success(f"已保存 {len(std_tpl.answers)} 题标准答案")
                    st.rerun()

            # 显示未识别题目的调试信息
            if hasattr(std_tpl, '_debug_samples') and std_tpl._debug_samples:
                unrecognized = [d for d in std_tpl._debug_samples if d["answer"] is None]
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
                        for d in std_tpl._debug_samples[:5]:
                            opts_str = " | ".join(f"{o}={v}" for o, v in d["opts"].items())
                            lines.append(
                                f"Q{d['q']:>2}: ans={d['answer']} best={d['best_opt']}={d['best_val']} "
                                f"gap={d['gap']} | {opts_str}"
                            )
                        st.code("\n".join(lines), language=None)
                        if len(std_tpl._debug_samples) > 5:
                            st.caption(f"... 共 {len(std_tpl._debug_samples)} 题")

        # 预览气泡覆盖
        with st.expander("查看气泡覆盖预览"):
            vis = timg.copy()
            # 画列框边界和内部网格线(2026-06-08 适配 option_axis)
            for cfg in st.session_state.template_column_configs:
                x1, y1, x2, y2 = cfg["x1"], cfg["y1"], cfg["x2"], cfg["y2"]
                option_axis = cfg.get("option_axis", "x")
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                if option_axis == "x":
                    # 标准:题号在 y 方向,画水平分隔线
                    row_h = (y2 - y1) / cfg["num_q"]
                    for qi in range(1, cfg["num_q"]):
                        ly = int(y1 + qi * row_h)
                        cv2.line(vis, (x1, ly), (x2, ly), (0, 255, 255), 1)
                else:
                    # 横排题:题号在 x 方向,画垂直分隔线
                    col_w = (x2 - x1) / cfg["num_q"]
                    for qi in range(1, cfg["num_q"]):
                        lx = int(x1 + qi * col_w)
                        cv2.line(vis, (lx, y1), (lx, y2), (0, 255, 255), 1)
            # 画气泡采样点
            for b in std_tpl.bubbles:
                cv2.circle(vis, (b["x"], b["y"]), max(4, b["w"] // 2), (0, 255, 0), 2)
                cv2.line(vis, (b["x"]-3, b["y"]), (b["x"]+3, b["y"]), (0, 0, 255), 2)
                cv2.line(vis, (b["x"], b["y"]-3), (b["x"], b["y"]+3), (0, 0, 255), 2)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                     caption="黄色=列框与分隔线 | 绿色=采样窗口 | 红色十字=采样中心", use_column_width=True)

# ---------- Tab 2: 批量处理 ----------
with tab2:
    st.header("批量处理答题卡")
    st.caption("使用标定的截取区域裁剪图片，并用标准模板识别选择题答案")

    has_a = bool(st.session_state.manual_regions_a)
    has_b = bool(st.session_state.manual_regions_b)
    has_template = st.session_state.standard_template is not None

    if not has_a and not has_b:
        st.warning("请先在「模板与参考」页面标定截取区域")
    elif not has_template:
        st.warning("请先在「模板与参考」页面第三步「标准模板标定」中生成标准模板")
    else:
        col_status = st.columns(3)
        with col_status[0]:
            st.success(f"A面 {len(st.session_state.manual_regions_a)} 个区域")
        with col_status[1]:
            st.success(f"B面 {len(st.session_state.manual_regions_b)} 个区域")
        with col_status[2]:
            st.success(f"标准模板 {len(st.session_state.template_answers)} 题答案")

        # mc_side 状态显示 + 错配警告(防止用户没切 radio 导致 0 识别)
        mc_side = st.session_state.mc_side
        if mc_side == "A" and not has_a and has_b:
            st.error("❌ Tab1 设置了「选择题在 A 面」,但 A 面 0 个区域 — 识别会全部失败!请回 Tab1 「2.标准模板标定」顶部切到 **B 面**。")
        elif mc_side == "B" and not has_b and has_a:
            st.error("❌ Tab1 设置了「选择题在 B 面」,但 B 面 0 个区域 — 识别会全部失败!请回 Tab1 「2.标准模板标定」顶部切到 **A 面**。")
        else:
            st.info(f"📍 选择题识别目标: **{mc_side}面**  (Tab1 「2.标准模板标定」顶部可切换)")

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

            if st.button("开始处理", type="primary"):
                std_tpl =st.session_state.standard_template
                output_dir = os.path.join("output", "batch")
                results = []
                crop_summary = []

                bar = st.progress(0)
                status = st.empty()
                total = len(valid) + len(single_a) + len(single_b)
                processed = [0]  # 用列表包装，避免 nonlocal 问题

                def _process_single(key, file_a, file_b, recognize_side):
                    """处理单张答题卡。
                    recognize_side="A": 读 file_a 识别,file_b 仍可裁剪
                    recognize_side="B": 读 file_b 识别,file_a 仍可裁剪
                    两面裁剪都按 manual_regions_a/b 各自跑(若有)。
                    """
                    processed[0] += 1
                    status.info(f"处理中 [{processed[0]}/{total}]: {key}")

                    # 读识别目标图
                    target_file = file_a if recognize_side == "A" else file_b
                    if target_file is None:
                        # 当前面没有图(比如单 A + mc_side=B),无法识别
                        return None
                    tf = target_file; tf.seek(0)
                    img_target = cv2.imdecode(np.frombuffer(tf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img_target is None:
                        return None

                    # 1. 标准模板识别选择题 — 按 recognize_side 选图
                    recognizer = make_recognizer("standard", standard_template=std_tpl)
                    result = recognizer.recognize(
                        img_target,
                        RecognizeContext(standard_answers=st.session_state.standard_answers),
                    )
                    r = result.to_legacy_dict()        # 转回 dict 形态,下游代码 0 改动
                    r["_key"] = key
                    r["_file_a"] = file_a.name if file_a else ""
                    r["_file_b"] = file_b.name if file_b else ""
                    r["_mc_side"] = recognize_side       # 标记用了哪面识别

                    # 1b. 生成识别预览图（气泡采样点叠加在识别目标图上）
                    os.makedirs(output_dir, exist_ok=True)
                    preview = img_target.copy()
                    h_p, w_p = preview.shape[:2]
                    gt_img = std_tpl.image
                    gt_h, gt_w = gt_img.shape[:2]
                    for b in std_tpl.bubbles:
                        q, opt = b["q"], b["opt"]
                        # 坐标从标准模板尺寸缩放到实际图像尺寸
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
                    preview_path = os.path.join(output_dir, f"{key}_template_preview_{recognize_side}.png")
                    _, png_buf = cv2.imencode(".png", preview)
                    with open(preview_path, "wb") as pf:
                        pf.write(png_buf)
                    r["_preview_path"] = preview_path

                    # 2. 计分（与标准答案对比）
                    correct = 0
                    total_q = r["total"]
                    for ans in r["answers"].values():
                        if ans.get("correct") is True:
                            correct += 1
                    r["_score"] = correct
                    r["_total"] = total_q

                    # 白卷检测：没有任何一题被确认识别为 single/multi (answer 全空)
                    # 旧版"识别率<50%" 太宽松,容易把浅填涂/扫描噪声多的卷子漏判
                    # 新版: 一题都没识别出来才判白卷(0%严格口径)
                    total_q = r["total"]
                    identified_count = sum(1 for ans in r["answers"].values()
                                            if ans.get("answer") is not None)
                    r["_is_blank"] = identified_count == 0 and total_q > 0

                    # 全卷异常检测: 识别率 < 30% 且已识别题正确率 < 10% → 极可能是白卷/扫描异常
                    # 防御"空白卷+扫描噪声 → 误识别为 single"的场景
                    # 此时把"识别错"的答案降级为 uncertain (让人工核对),并判白卷
                    # 2026-06-08 调整: 旧条件"识别出 ≥3 题但正确率 < 10%"会误伤"故意错填"的卡
                    # 如 OMR0002 19A: 10 题都识别为 single 但答案和 gold 完全不同(可能故意错填),
                    # 旧逻辑会把它降级为白卷+10 题全 uncertain,用户看不到 19A 的真实填涂。
                    # 新条件加识别率门槛(< 30%),放过"全题都识别但全错"的卡,只对"几乎没识别出来"
                    # 的扫描异常卡降级
                    correct_count = sum(1 for ans in r["answers"].values()
                                        if ans.get("correct") is True)
                    if (total_q >= 3 and identified_count > 0
                            and identified_count < total_q * 0.3
                            and correct_count / identified_count < 0.1):
                        for ans in r["answers"].values():
                            if ans.get("answer") is not None and ans.get("correct") is not True:
                                ans["status"] = "uncertain"
                                ans["answer"] = None
                                ans["correct"] = None
                        r["_is_blank"] = True
                        r["_score"] = 0

                    if r["_is_blank"]:
                        r["_score"] = 0

                    # 3. 截取区域裁剪（A面）— 独立读图,与识别哪面无关
                    if has_a and file_a is not None:
                        fa2 = file_a; fa2.seek(0)
                        img_a = cv2.imdecode(np.frombuffer(fa2.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img_a is not None:
                            crops_a = CardProcessor.crop_by_regions(
                                img_a, st.session_state.manual_regions_a,
                                output_dir, f"{key}_A",
                                ref_size=st.session_state.ref_image_size_a)
                            for c in crops_a:
                                crop_summary.append({"key": key, "文件": file_a.name, "面别": "A", "区域": c["name"], "路径": c["path"]})

                    # 4. 截取区域裁剪（B面）
                    if has_b and file_b is not None:
                        fb2 = file_b; fb2.seek(0)
                        img_b = cv2.imdecode(np.frombuffer(fb2.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img_b is not None:
                            crops_b = CardProcessor.crop_by_regions(
                                img_b, st.session_state.manual_regions_b,
                                output_dir, f"{key}_B",
                                ref_size=st.session_state.ref_image_size_b)
                            for c in crops_b:
                                crop_summary.append({"key": key, "文件": file_b.name, "面别": "B", "区域": c["name"], "路径": c["path"]})

                    return r

                # 选择题所在面 — 决定 valid/single_a/single_b 各分支是否走识别
                mc_side = st.session_state.mc_side

                # 处理A+B配对 — 始终按 mc_side 识别对应面
                for key, files in valid.items():
                    r = _process_single(key, files["A"], files["B"], mc_side)
                    if r:
                        results.append(r)
                    bar.progress(int(processed[0] / total * 100))

                # 处理单A面 — mc_side=A 时才识别,否则仅裁剪
                for key, files in single_a.items():
                    if mc_side == "A":
                        r = _process_single(key, files["A"], None, "A")
                        if r:
                            results.append(r)
                    else:
                        # 仅裁剪 A 面
                        processed[0] += 1
                        status.info(f"处理中 [{processed[0]}/{total}]: {key} (仅A面裁剪)")
                        fa = files["A"]; fa.seek(0)
                        img_a = cv2.imdecode(np.frombuffer(fa.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img_a is not None and has_a:
                            crops_a = CardProcessor.crop_by_regions(
                                img_a, st.session_state.manual_regions_a,
                                output_dir, f"{key}_A",
                                ref_size=st.session_state.ref_image_size_a)
                            for c in crops_a:
                                crop_summary.append({"key": key, "文件": files["A"].name, "面别": "A", "区域": c["name"], "路径": c["path"]})
                    bar.progress(int(processed[0] / total * 100))

                # 处理单B面 — mc_side=B 时才识别,否则仅裁剪
                for key, files in single_b.items():
                    if mc_side == "B":
                        r = _process_single(key, None, files["B"], "B")
                        if r:
                            results.append(r)
                    else:
                        # 仅裁剪 B 面 (原行为)
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

        # 持久渲染: 切到 Tab3 再回来时也保留 (之前在 button 块内,刷新就消失)
        if st.session_state.results:
            st.divider()
            st.subheader("📊 识别结果摘要")
            rows = []
            for r in st.session_state.results:
                total_q = r["total"]
                # 统计口径与 Tab3 + 白卷判定一致: answer 非空 = 已答
                answered = sum(1 for ans in r["answers"].values()
                               if ans.get("answer") is not None)
                rate = answered / total_q * 100 if total_q > 0 else 0
                is_blank = r.get("_is_blank", False) or (answered == 0 and total_q > 0)
                rows.append({
                    "学生/文件": r["_key"],
                    "识别题数": f"{answered}/{total_q}",
                    "识别率": f"{rate:.0f}%",
                    "状态": "白卷" if is_blank else "正常",
                    "得分": f"{r.get('_score', 0)}/{r.get('_total', 0)}",
                    "漏涂": r.get("empty_count", 0),
                    "多选": r.get("multi_count", 0),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            blank_count = sum(1 for row in rows if row["状态"] == "白卷")
            if blank_count > 0:
                st.warning(f"检测到 {blank_count} 份白卷（系统自动判定的）")

            if st.session_state.crop_results:
                with st.expander(f"裁剪详情（{len(st.session_state.crop_results)} 个区域）"):
                    st.dataframe(pd.DataFrame(st.session_state.crop_results), use_container_width=True)

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

            # 标准模板识别预览图
            preview_path = result.get("_preview_path")
            if preview_path and os.path.exists(preview_path):
                with st.expander("🔍 标准模板识别预览（绿色=已识别选项，灰色=未识别，蓝色=多选）", expanded=False):
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
                std = st.session_state.template_answers.get(q, "")
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
                key="template_result_editor",
            )

            new_corr = {}
            for _, row in edited.iterrows():
                if row["人工修正"] and row["人工修正"].strip():
                    new_corr[int(row["题号"])] = row["人工修正"].strip().upper()
            st.session_state.manual_corrections[sid] = new_corr

            # 基于人工修正重新计分
            template_ans = st.session_state.template_answers
            # 构造 effective_answers: 人工修正优先,否则用识别器结果
            effective_answers = {}
            for q, std_ans in template_ans.items():
                corrected = new_corr.get(q)
                if corrected:
                    effective_answers[q] = {"answer": corrected, "status": "single"}
                else:
                    ans_info = result["answers"].get(q, {})
                    effective_answers[q] = {
                        "answer": ans_info.get("answer"),
                        "status": ans_info.get("status", "empty"),
                    }
            score_result = calc_total_score(effective_answers, template_ans, ScoringConfig())
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
