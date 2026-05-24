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
        "process_mode": "模板识别模式",  # 或 "手动区域裁剪模式"
        "custom_bubbles": [],           # (已移除UI，保留兼容)
        "custom_bubbles_img_size": None,
        "paper_layouts": {},          # 保存的选择题版式配置 {name: {...}}
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

# ---------- Tab 1: 模板与空白参考 + 手动标定区域 ----------
with tab1:
    # ===== 原有的空白参考上传 =====
    st.header("1. 上传空白答题卡作为识别基准")
    st.info("差分法需要空白答题卡做对比，请上传未填涂的A面和B面空白答题卡（用于提高OMR识别准确率）")
    
    c1, c2 = st.columns(2)
    with c1:
        blank_a_file = st.file_uploader("A面空白答题卡", type=["jpg","jpeg","png"], key="blank_a_up")
        if blank_a_file:
            bytes_a = np.asarray(bytearray(blank_a_file.read()), dtype=np.uint8)
            st.session_state.blank_a = cv2.imdecode(bytes_a, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(st.session_state.blank_a, cv2.COLOR_BGR2RGB), caption="A面空白模板")
    
    with c2:
        blank_b_file = st.file_uploader("B面空白答题卡", type=["jpg","jpeg","png"], key="blank_b_up")
        if blank_b_file:
            bytes_b = np.asarray(bytearray(blank_b_file.read()), dtype=np.uint8)
            st.session_state.blank_b = cv2.imdecode(bytes_b, cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(st.session_state.blank_b, cv2.COLOR_BGR2RGB), caption="B面空白模板")
    
    if st.session_state.processor and st.session_state.blank_a is not None:
        if st.button("设置空白参考"):
            st.session_state.processor.set_blank_ref(st.session_state.blank_a, st.session_state.blank_b)
            st.success("✅ 空白参考已设置，OMR将使用差分法识别")
    
    st.divider()
    
    # ===== 手动标定截取区域 =====
    st.header("2. 手动标定截取区域（用于批量裁剪）")
    st.info("配置A面和B面的截取区域坐标，批量裁剪时会按比例缩放到实际图片尺寸。支持「可视化画框」和「纯坐标输入」两种模式。")
    
    calib_mode = st.radio("标定方式", ["🖼️ 上传图片画框", "✏️ 直接输入坐标"], horizontal=True, key="calib_mode")
    
    def render_side(col, is_a, calib_mode):
        with col:
            side_label = "A面" if is_a else "B面"
            st.subheader(f"📄 {side_label}")
            regions_key = "manual_regions_a" if is_a else "manual_regions_b"
            regions_list = st.session_state[regions_key]
            key_prefix = "ra" if is_a else "rb"
            size_key = "ref_image_size_a" if is_a else "ref_image_size_b"
            
            ref_img = None
            canvas_scale = 1.0
            
            if calib_mode == "🖼️ 上传图片画框":
                if is_a and st.session_state.blank_a is not None:
                    ref_img = st.session_state.blank_a.copy()
                elif not is_a and st.session_state.blank_b is not None:
                    ref_img = st.session_state.blank_b.copy()
                else:
                    ref_img = None
                
                if ref_img is not None:
                    h, w = ref_img.shape[:2]
                    st.session_state[size_key] = (w, h)
                    st.write(f"📐 图片尺寸：{w} × {h}")
                    
                    MAX_CANVAS_WIDTH = 700
                    canvas_scale = MAX_CANVAS_WIDTH / w
                    display_w = MAX_CANVAS_WIDTH
                    display_h = int(h * canvas_scale)
                    
                    bg_image = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
                    bg_image = bg_image.convert("RGBA")
                    bg_image = bg_image.resize((display_w, display_h))
                    
                    st.markdown("**🖱️ 拖拽画出截取区域：**")
                    
                    canvas_ver_key = f"canvas_version_{key_prefix}"
                    canvas_version = st.session_state.get(canvas_ver_key, 0)
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.2)",
                        stroke_width=2,
                        stroke_color="#FF0000",
                        background_image=bg_image,
                        height=display_h,
                        width=display_w,
                        drawing_mode="rect",
                        key=f"canvas_{side_label}_{canvas_version}",
                        update_streamlit=True,
                    )
                    
                    if canvas_result.json_data is not None:
                        rects = [obj for obj in canvas_result.json_data.get("objects", []) if obj.get("type") == "rect"]
                        if rects:
                            st.write(f"检测到 **{len(rects)}** 个框")
                            col_add, col_clear = st.columns([1, 1])
                            with col_add:
                                if st.button("➕ 添加", key=f"btn_add_canvas_{key_prefix}"):
                                    for obj in rects:
                                        x1 = int(obj["left"] / canvas_scale)
                                        y1 = int(obj["top"] / canvas_scale)
                                        x2 = int((obj["left"] + obj["width"]) / canvas_scale)
                                        y2 = int((obj["top"] + obj["height"]) / canvas_scale)
                                        regions_list.append({
                                            "name": f"区域{len(regions_list)+1}",
                                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                            "type": "非选择题"
                                        })
                                    st.session_state[regions_key] = regions_list
                                    st.session_state[canvas_ver_key] = canvas_version + 1
                                    st.rerun()
                            with col_clear:
                                if st.button("🗑️ 清空", key=f"btn_clear_canvas_{key_prefix}"):
                                    st.session_state[canvas_ver_key] = canvas_version + 1
                                    st.rerun()
                        else:
                            st.caption("请在图片上拖拽画出矩形框")
                else:
                    st.warning("⚠️ 请先在上方的「上传空白答题卡」区域上传本面的空白答题卡，或切换到「直接输入坐标」模式")
            
            else:  # 直接输入坐标
                col_sw, col_sh = st.columns(2)
                with col_sw:
                    default_w = st.session_state.ref_image_size_a[0] if (is_a and st.session_state.ref_image_size_a) else (st.session_state.ref_image_size_b[0] if st.session_state.ref_image_size_b else 1237)
                    set_w = st.number_input("宽", min_value=1, value=default_w, key=f"set_w_{key_prefix}")
                with col_sh:
                    default_h = st.session_state.ref_image_size_a[1] if (is_a and st.session_state.ref_image_size_a) else (st.session_state.ref_image_size_b[1] if st.session_state.ref_image_size_b else 1741)
                    set_h = st.number_input("高", min_value=1, value=default_h, key=f"set_h_{key_prefix}")
                st.session_state[size_key] = (int(set_w), int(set_h))
                
                st.markdown("**添加新区域：**")
                c_in = st.columns([2, 1.5, 1, 1, 1, 1])
                with c_in[0]:
                    in_name = st.text_input("名称", value=f"区域{len(regions_list)+1}", key=f"in_name_{key_prefix}")
                with c_in[1]:
                    in_type = st.selectbox("类型", ["非选择题", "选择题", "个人信息"], key=f"in_type_{key_prefix}")
                with c_in[2]:
                    in_x1 = st.number_input("x1", min_value=0, value=0, key=f"in_x1_{key_prefix}")
                with c_in[3]:
                    in_y1 = st.number_input("y1", min_value=0, value=0, key=f"in_y1_{key_prefix}")
                with c_in[4]:
                    in_x2 = st.number_input("x2", min_value=0, value=100, key=f"in_x2_{key_prefix}")
                with c_in[5]:
                    in_y2 = st.number_input("y2", min_value=0, value=100, key=f"in_y2_{key_prefix}")
                if st.button("➕ 添加", key=f"btn_add_manual_{key_prefix}"):
                    regions_list.append({
                        "name": in_name,
                        "x1": int(in_x1), "y1": int(in_y1),
                        "x2": int(in_x2), "y2": int(in_y2),
                        "type": in_type
                    })
                    st.session_state[regions_key] = regions_list
                    st.rerun()
            
            # 区域列表
            st.divider()
            st.markdown("**📌 已添加的区域**")
            
            current_regions = []
            for i in range(len(regions_list)):
                region = regions_list[i]
                name = st.session_state.get(f"{key_prefix}_{i}_name", region["name"])
                x1 = st.session_state.get(f"{key_prefix}_{i}_x1", region["x1"])
                y1 = st.session_state.get(f"{key_prefix}_{i}_y1", region["y1"])
                x2 = st.session_state.get(f"{key_prefix}_{i}_x2", region["x2"])
                y2 = st.session_state.get(f"{key_prefix}_{i}_y2", region["y2"])
                rtype = st.session_state.get(f"{key_prefix}_{i}_type", region.get("type", "非选择题"))
                current_regions.append({"name": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "type": rtype})
            
            if current_regions:
                for i, region in enumerate(current_regions):
                    cols = st.columns([2, 1.5, 1, 1, 1, 1, 1])
                    with cols[0]:
                        st.text_input("名称", value=region["name"], key=f"{key_prefix}_{i}_name")
                    with cols[1]:
                        st.selectbox("类型", ["非选择题", "选择题", "个人信息"], 
                                     index=["非选择题", "选择题", "个人信息"].index(region["type"]),
                                     key=f"{key_prefix}_{i}_type")
                    with cols[2]:
                        st.number_input("x1", value=region["x1"], min_value=0, key=f"{key_prefix}_{i}_x1")
                    with cols[3]:
                        st.number_input("y1", value=region["y1"], min_value=0, key=f"{key_prefix}_{i}_y1")
                    with cols[4]:
                        st.number_input("x2", value=region["x2"], min_value=0, key=f"{key_prefix}_{i}_x2")
                    with cols[5]:
                        st.number_input("y2", value=region["y2"], min_value=0, key=f"{key_prefix}_{i}_y2")
                    with cols[6]:
                        st.write("")
                        st.write("")
                        if st.button("❌", key=f"{key_prefix}_del_{i}"):
                            regions_list.pop(i)
                            st.session_state[regions_key] = regions_list
                            for suffix in ["name", "type", "x1", "y1", "x2", "y2"]:
                                k = f"{key_prefix}_{i}_{suffix}"
                                if k in st.session_state:
                                    del st.session_state[k]
                            st.rerun()
            else:
                st.caption("暂无区域")
            
            # 实时同步
            synced = []
            for i in range(len(regions_list)):
                synced.append({
                    "name": st.session_state.get(f"{key_prefix}_{i}_name", regions_list[i]["name"]),
                    "x1": st.session_state.get(f"{key_prefix}_{i}_x1", regions_list[i]["x1"]),
                    "y1": st.session_state.get(f"{key_prefix}_{i}_y1", regions_list[i]["y1"]),
                    "x2": st.session_state.get(f"{key_prefix}_{i}_x2", regions_list[i]["x2"]),
                    "y2": st.session_state.get(f"{key_prefix}_{i}_y2", regions_list[i]["y2"]),
                    "type": st.session_state.get(f"{key_prefix}_{i}_type", regions_list[i].get("type", "非选择题")),
                })
            st.session_state[regions_key] = synced
            
            # 预览
            st.write("---")
            st.markdown("**🎨 预览**")
            img_size = st.session_state.get(size_key)
            if img_size and synced:
                if ref_img is not None:
                    vis = ref_img.copy()
                else:
                    vis = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
                h, w = vis.shape[:2]
                type_colors = {
                    "非选择题": (0, 255, 0),
                    "选择题": (255, 0, 0),
                    "个人信息": (0, 0, 255),
                }
                for idx, region in enumerate(synced):
                    color = type_colors.get(region["type"], (0,255,0))
                    x1 = max(0, min(w, region["x1"]))
                    y1 = max(0, min(h, region["y1"]))
                    x2 = max(0, min(w, region["x2"]))
                    y2 = max(0, min(h, region["y2"]))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    label = f"{region['name']}({region['type'][:2]})"
                    cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"{side_label}预览", use_column_width=True)
            else:
                st.caption("添加区域后显示预览")
    
    col_a, col_b = st.columns(2)
    render_side(col_a, True, calib_mode)
    render_side(col_b, False, calib_mode)
    
    # 导出/导入配置
    st.divider()
    col_exp, col_imp = st.columns(2)
    with col_exp:
        export_data = {
            "mode": "manual_regions_v2",
            "image_size_a": {"w": st.session_state.ref_image_size_a[0], "h": st.session_state.ref_image_size_a[1]} if st.session_state.ref_image_size_a else None,
            "image_size_b": {"w": st.session_state.ref_image_size_b[0], "h": st.session_state.ref_image_size_b[1]} if st.session_state.ref_image_size_b else None,
            "regions_a": st.session_state.manual_regions_a,
            "regions_b": st.session_state.manual_regions_b,
        }
        st.download_button(
            label="📥 导出区域配置(JSON)",
            data=json.dumps(export_data, ensure_ascii=False, indent=2),
            file_name="manual_regions.json",
            mime="application/json"
        )
    with col_imp:
        imported = st.file_uploader("导入区域配置(JSON)", type=["json"], key="import_regions")
        if imported:
            try:
                data = json.loads(imported.read().decode("utf-8"))
                if "regions_a" in data:
                    st.session_state.manual_regions_a = data["regions_a"]
                if "regions_b" in data:
                    st.session_state.manual_regions_b = data["regions_b"]
                # 兼容旧格式
                if "regions" in data and "regions_a" not in data:
                    st.session_state.manual_regions_a = data["regions"]
                st.success("✅ 已导入区域配置")
                st.rerun()
            except Exception as e:
                st.error(f"导入失败: {e}")
    
    st.divider()
    
    # ===== 原有的模板预览 =====
    st.header("3. 模板预览")
    if st.session_state.processor:
        proc = st.session_state.processor
        tmpl = proc.template
        
        preview_file = st.file_uploader("上传任意答题卡预览标定效果", type=["jpg","jpeg","png"], key="preview")
        if preview_file:
            bytes_p = np.asarray(bytearray(preview_file.read()), dtype=np.uint8)
            img = cv2.imdecode(bytes_p, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            vis = img.copy()
            
            # 画选项框
            for b in tmpl["pages"]["A"].get("bubbles", []):
                bx, by = proc.scale_coords(b["x"], b["y"], w, h)
                bw = max(6, int(b["w"] * w / proc.ref_w))
                cv2.circle(vis, (bx, by), bw, (0,255,0), 1)
            
            # 画主观题区域
            for qn, coords in tmpl["pages"]["A"].get("subjective", {}).items():
                x1, y1 = proc.scale_coords(coords["x1"], coords["y1"], w, h)
                x2, y2 = proc.scale_coords(coords["x2"], coords["y2"], w, h)
                cv2.rectangle(vis, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.putText(vis, qn, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            
            # 画手动标定区域（A面）
            if st.session_state.manual_regions_a and st.session_state.ref_image_size_a:
                ref_w_a, ref_h_a = st.session_state.ref_image_size_a
                scale_x_a = w / ref_w_a
                scale_y_a = h / ref_h_a
                type_colors = {
                    "非选择题": (0, 255, 255),   # 青色
                    "选择题": (255, 0, 255),     # 紫色
                    "个人信息": (255, 255, 0),   # 黄色
                }
                for region in st.session_state.manual_regions_a:
                    rx1 = int(region["x1"] * scale_x_a)
                    ry1 = int(region["y1"] * scale_y_a)
                    rx2 = int(region["x2"] * scale_x_a)
                    ry2 = int(region["y2"] * scale_y_a)
                    rtype = region.get("type", "非选择题")
                    color = type_colors.get(rtype, (0, 255, 255))
                    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), color, 3)
                    label = f"{region['name']}({rtype[:2]})"
                    cv2.putText(vis, label, (rx1, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="绿色=选项框中心，蓝色=模板主观题区域，青/紫/黄色=手动标定区域", use_column_width=True)
    else:
        st.warning("请先加载模板")
    
    # ===== 选择题版式配置（半自动网格） =====
    st.divider()
    st.header("4. 选择题版式配置（半自动网格）")
    st.info("新版式试卷无需逐个点选选项框。在图片上画出每列的大致区域，输入行列数，系统自动生成网格并判断填涂。")
    
    layout_file = st.file_uploader("上传选择题区域样图（用于配置版式）", type=["jpg","jpeg","png"], key="layout_sample")
    
    if layout_file:
        bytes_l = np.asarray(bytearray(layout_file.read()), dtype=np.uint8)
        layout_img = cv2.imdecode(bytes_l, cv2.IMREAD_COLOR)
        lh, lw = layout_img.shape[:2]
        st.write(f"📐 图片尺寸: {lw} × {lh}")
        
        # Canvas 显示尺寸
        MAX_CANVAS_WIDTH = 800
        canvas_scale = MAX_CANVAS_WIDTH / lw
        display_w = MAX_CANVAS_WIDTH
        display_h = int(lh * canvas_scale)
        
        bg_image = Image.fromarray(cv2.cvtColor(layout_img, cv2.COLOR_BGR2RGB))
        bg_image = bg_image.convert("RGBA")
        bg_image = bg_image.resize((display_w, display_h))
        
        st.markdown("**🖱️ 在下方图片上拖拽画出各列的矩形区域（从左到右画，红色框）：**")
        st.caption("画完所有列后，点击「提取画框」；如需重画请点击「清空画布」")
        
        canvas_ver = st.session_state.get("layout_canvas_ver", 0)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=bg_image,
            height=display_h,
            width=display_w,
            drawing_mode="rect",
            key=f"layout_canvas_{canvas_ver}",
            update_streamlit=True,
        )
        
        col_extract, col_clear = st.columns([1, 1])
        with col_extract:
            extract_clicked = st.button("➕ 提取画框并配置参数", key="layout_extract")
        with col_clear:
            if st.button("🗑️ 清空画布", key="layout_clear_canvas"):
                st.session_state.layout_canvas_ver = canvas_ver + 1
                if "layout_draft_boxes" in st.session_state:
                    del st.session_state.layout_draft_boxes
                st.rerun()
        
        col_rec = st.columns([1])[0]
        with col_rec:
            if st.button("✨ 使用推荐框(自动)", key="use_recommended"):
                st.session_state.layout_draft_boxes = [
                    {'x1': 50, 'y1': 28, 'x2': 160, 'y2': 132},
                    {'x1': 240, 'y1': 29, 'x2': 350, 'y2': 135},
                    {'x1': 430, 'y1': 25, 'x2': 670, 'y2': 138},
                    {'x1': 50, 'y1': 155, 'x2': 160, 'y2': 264},
                    {'x1': 240, 'y1': 156, 'x2': 350, 'y2': 266},
                    {'x1': 430, 'y1': 157, 'x2': 560, 'y2': 262},
                    {'x1': 50, 'y1': 285, 'x2': 160, 'y2': 392},
                    {'x1': 240, 'y1': 287, 'x2': 350, 'y2': 394},
                    {'x1': 430, 'y1': 275, 'x2': 560, 'y2': 340},
                ]
                st.rerun()
        
        if extract_clicked and canvas_result.json_data is not None:
            rects = [obj for obj in canvas_result.json_data.get("objects", []) if obj.get("type") == "rect"]
            if rects:
                box_list = []
                for obj in rects:
                    x1 = int(obj["left"] / canvas_scale)
                    y1 = int(obj["top"] / canvas_scale)
                    x2 = int((obj["left"] + obj["width"]) / canvas_scale)
                    y2 = int((obj["top"] + obj["height"]) / canvas_scale)
                    box_list.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
                # 不再自动排序，保持用户画框的原始顺序（Canvas返回的顺序）
                # 用户应按 1→2→3... 的视觉顺序依次画框
                st.session_state.layout_draft_boxes = box_list
                st.session_state.layout_canvas_ver = canvas_ver + 1
                st.rerun()
            else:
                st.warning("请先画出至少一个矩形框")
        
        # 已提取的框 → 参数配置
        if st.session_state.get("layout_draft_boxes"):
            boxes = st.session_state.layout_draft_boxes
            st.markdown(f"**✏️ 已提取 {len(boxes)} 个列框，请为每列配置参数：**")
            
            # 批量设置工具栏
            st.markdown("---")
            st.caption("批量设置工具：先统一设一个默认值，再单独微调个别列")
            bc1, bc2, bc3, bc4 = st.columns([1,1,1,1])
            with bc1:
                batch_q = st.number_input("默认题目数", min_value=1, value=5, key="batch_nq")
            with bc2:
                batch_o = st.number_input("默认选项数", min_value=2, value=4, key="batch_no")
            with bc3:
                if st.button("应用到全部", key="batch_apply"):
                    for i in range(len(boxes)):
                        st.session_state[f"ld_nq_{i}"] = int(batch_q)
                        st.session_state[f"ld_no_{i}"] = int(batch_o)
                    st.rerun()
            with bc4:
                if st.button("恢复默认", key="batch_reset"):
                    for i in range(len(boxes)):
                        st.session_state[f"ld_sq_{i}"] = i*5+1
                        st.session_state[f"ld_nq_{i}"] = 5
                        st.session_state[f"ld_no_{i}"] = 3
                    st.rerun()
            st.markdown("---")
            
            configs = []
            for i, box in enumerate(boxes):
                st.markdown(f"**第 {i+1} 列** — 框范围: ({box['x1']},{box['y1']}) ~ ({box['x2']},{box['y2']})")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    sq = st.number_input("起始题号", min_value=1, value=i*5+1, key=f"ld_sq_{i}")
                with c2:
                    nq = st.number_input("题目数", min_value=1, value=5, key=f"ld_nq_{i}")
                with c3:
                    no = st.number_input("选项数", min_value=2, value=3, key=f"ld_no_{i}")
                with c4:
                    thr = st.number_input("填涂阈值", min_value=0.05, max_value=0.80, value=0.25, step=0.05, key=f"ld_thr_{i}")
                configs.append({
                    "start_q": int(sq), "num_q": int(nq), "num_options": int(no),
                    "threshold": float(thr),
                    **box
                })
            
            # 先显示配置摘要，方便核对
            summary = []
            for cfg in configs:
                summary.append({
                    "起始题": cfg["start_q"],
                    "题数": cfg["num_q"],
                    "选项数": cfg["num_options"],
                    "阈值": cfg["threshold"],
                    "框坐标": f"({cfg['x1']},{cfg['y1']})~({cfg['x2']},{cfg['y2']})"
                })
            st.dataframe(summary, hide_index=True)
            
            # ===== 实时预览与识别 =====
            st.markdown("---")
            st.markdown("**👁️ 实时预览与识别**（修改上方参数后自动更新）")
            st.caption("每列使用各自独立的填涂阈值，可在上方表格中设置")
            
            vis = layout_img.copy()
            all_bubbles = []
            for cfg in configs:
                sx, sy, ex, ey = cfg["x1"], cfg["y1"], cfg["x2"], cfg["y2"]
                nq, no, sq = cfg["num_q"], cfg["num_options"], cfg["start_q"]
                cv2.rectangle(vis, (sx, sy), (ex, ey), (255, 0, 0), 2)
                
                row_h = (ey - sy) / nq
                col_w = (ex - sx) / no
                
                for qi in range(nq):
                    qn = sq + qi
                    cy = int(sy + qi * row_h + row_h / 2)
                    for oi in range(no):
                        cx = int(sx + oi * col_w + col_w / 2)
                        letter = chr(ord('A') + oi)
                        r = max(3, int(min(col_w, row_h) * 0.35))
                        cv2.circle(vis, (cx, cy), r, (0, 255, 0), 1)
                        cv2.putText(vis, letter, (cx + 4, cy + 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                        all_bubbles.append({
                            "q": qn, "opt": letter,
                            "x": cx, "y": cy,
                            "w": max(8, int(col_w * 0.5)),
                            "h": max(8, int(row_h * 0.5)),
                            "threshold": cfg["threshold"]
                        })

            # 实时测试识别 — 改进版：Otsu + 相对差异 + 多半径 + fallback
            gray = cv2.cvtColor(layout_img, cv2.COLOR_BGR2GRAY)
            rec_results = {}
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            def detect_bubble(bx, by, bw, bh):
                """固定小半径中心检测 + 暗度评分，过滤印刷圆环"""
                # 固定用小半径(0.25)，只看气泡正中心，过滤边缘印刷线
                roi_w = max(4, int(bw * 0.25))
                roi_h = max(4, int(bh * 0.25))
                x1 = max(0, bx - roi_w // 2)
                y1 = max(0, by - roi_h // 2)
                x2 = min(lw, bx + roi_w // 2)
                y2 = min(lh, by + roi_h // 2)
                roi = blurred[y1:y2, x1:x2]
                if roi.size == 0:
                    return 0, 255, 999
                # 用暗度作为主要指标
                mean_g = float(np.mean(roi))
                std_g = float(np.std(roi))
                # 暗度评分：空白气泡中心白(~228)→0分，完全填涂(~140)→1分
                darkness = max(0, (230 - mean_g) / 90)
                darkness = min(1.0, darkness)
                return darkness, mean_g, std_g

            # 第一遍：检测每个选项
            for b in all_bubbles:
                q, opt = b["q"], b["opt"]
                bx, by = b["x"], b["y"]
                bw = b["w"]
                bh = b["h"]
                darkness, mean_gray, std_g = detect_bubble(bx, by, bw, bh)
                if q not in rec_results:
                    rec_results[q] = {"filled": [], "darkness": {}, "scores": {}, "stds": {}}
                rec_results[q]["darkness"][opt] = round(darkness, 3)
                rec_results[q]["scores"][opt] = round(mean_gray, 1)
                rec_results[q]["stds"][opt] = round(std_g, 1)

            # 第二遍：判断 — 绝对暗度 + 相对差异
            # 判断 — 关键要求：
            # 1. best_score < 200：气泡中心真的变暗了（不是坐标偏离到黑块）
            # 2. rel_diff > 0.05：该选项明显比其他选项暗
            # 3. best_score 极低(<100)通常是坐标错误导致，视为无效
            for q, data in rec_results.items():
                scores = data["scores"]
                darkness = data["darkness"]
                sorted_opts = sorted(scores.items(), key=lambda x: x[1])
                if len(sorted_opts) >= 2:
                    best_opt, best_score = sorted_opts[0]
                    second_score = sorted_opts[1][1]
                    diff = second_score - best_score
                    rel_diff = diff / best_score if best_score > 0 else 0
                    best_darkness = darkness.get(best_opt, 0)
                    # 绝对暗度门槛：中心必须真的黑（best_score < 200）
                    # 相对差异门槛：提高到0.05减少误检
                    # 极低分通常是坐标错误，不接受
                    if 100 < best_score < 200 and best_darkness > 0.30 and rel_diff > 0.05:
                        data["filled"] = [best_opt]
                        data["rel_diff"] = round(rel_diff, 3)
                    else:
                        data["filled"] = []
                elif len(sorted_opts) == 1:
                    opt = list(scores.keys())[0]
                    d = darkness[opt]
                    s = list(scores.values())[0]
                    if 100 < s < 200 and d > 0.50:
                        data["filled"] = [opt]
                    else:
                        data["filled"] = []


            # 预览图：检测到的填涂用红色标记
            vis_bubbles = vis.copy()
            detected_per_q = {q: data["filled"] for q, data in rec_results.items()}
            for b in all_bubbles:
                qn, opt = b["q"], b["opt"]
                cx, cy = b["x"], b["y"]
                r = max(3, int(min(b["w"], b["h"]) * 0.4))
                is_filled = opt in detected_per_q.get(qn, [])
                color = (0, 0, 255) if is_filled else (0, 255, 0)  # 红=填涂，绿=未填涂
                cv2.circle(vis_bubbles, (cx, cy), r, color, -1)
                cv2.putText(vis_bubbles, opt, (cx + 4, cy + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            st.image(cv2.cvtColor(vis_bubbles, cv2.COLOR_BGR2RGB),
                     caption="绿色=未填涂，红色=已检测到填涂（共{}个选项框）".format(len(all_bubbles)), use_column_width=True)

            out_cols = st.columns(3)
            col_idx = 0
            for q in sorted(rec_results.keys()):
                with out_cols[col_idx % 3]:
                    filled = rec_results[q]["filled"]
                    scores = rec_results[q]["scores"]
                    darkness = rec_results[q]["darkness"]
                    rel_diff = rec_results[q].get("rel_diff", 0)
                    fs = ", ".join(filled) if filled else "未填涂"
                    st.write(f"**第 {q} 题**: {fs}")
                    st.caption("rel={:.3f} | {}".format(
                        rel_diff,
                        " | ".join(["{}{:.1f}(d{:.2f})".format(k, scores[k], darkness[k]) for k in sorted(scores.keys())])
                    ))
                col_idx += 1
            
            # 保存
            st.divider()
            name = st.text_input("版式名称（保存后可在批量处理中使用）", value="new_layout", key="layout_name_input")
            if st.button("💾 保存此版式"):
                if "paper_layouts" not in st.session_state:
                    st.session_state.paper_layouts = {}
                st.session_state.paper_layouts[name] = {
                    "image_size": {"w": lw, "h": lh},
                    "configs": configs,
                    "bubbles": all_bubbles
                }
                st.success(f"版式 '{name}' 已保存！共 {len(all_bubbles)} 个选项框。")
    
    # 显示已保存的版式
    if st.session_state.get("paper_layouts"):
        st.divider()
        st.markdown("**📋 已保存的版式：**")
        for ln, ld in st.session_state.paper_layouts.items():
            st.write(f"- `{ln}`: {len(ld['bubbles'])} 个选项框，图片尺寸 {ld['image_size']['w']}×{ld['image_size']['h']}")
        if st.button("🗑️ 清空所有版式"):
            st.session_state.paper_layouts = {}
            st.rerun()

# ---------- Tab 2: 批量处理 ----------
with tab2:
    st.header("批量处理答题卡")
    
    # 处理模式选择
    mode = st.radio(
        "选择处理模式",
        ["模板识别模式", "手动区域裁剪模式"],
        index=0 if st.session_state.process_mode == "模板识别模式" else 1,
        horizontal=True
    )
    st.session_state.process_mode = mode
    
    if mode == "模板识别模式":
        # ===== 原有流程 =====
        if st.session_state.processor is None:
            st.error("⚠️ 请先加载模板")
        else:
            st.info("请同时上传A面和B面图片（文件名需对应，如 `xxx01A.jpg` 和 `xxx01B.jpg`）")
            uploaded = st.file_uploader("批量上传", type=["jpg","jpeg","png"], accept_multiple_files=True, key="batch_template")
            
            if uploaded:
                # 配对
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
                
                valid = {k:v for k,v in pairs.items() if "A" in v and "B" in v}
                st.write(f"识别到 **{len(valid)}** 组有效答题卡（A+B配对）")
                
                col_run, col_thresh = st.columns([1,2])
                with col_thresh:
                    threshold = st.slider("OMR识别阈值", 0.02, 0.30, 0.10, 0.01, 
                                          help="差分比例阈值，越低越灵敏（可能误识），越高越严格（可能漏识）")
                with col_run:
                    st.write("")
                    st.write("")
                    run_btn = st.button("🚀 开始处理", type="primary")
                
                if run_btn:
                    proc = st.session_state.processor
                    results = []
                    bar = st.progress(0)
                    status = st.empty()
                    
                    for idx, (key, files) in enumerate(valid.items()):
                        status.info(f"处理中 [{idx+1}/{len(valid)}]: {key}")
                        
                        fa = files["A"]; fa.seek(0)
                        fb = files["B"]; fb.seek(0)
                        img_a = cv2.imdecode(np.asarray(bytearray(fa.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                        img_b = cv2.imdecode(np.asarray(bytearray(fb.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                        
                        if img_a is None or img_b is None:
                            continue
                        
                        result = proc.process_pair(
                            img_a, img_b, student_id=key,
                            manual_regions_a=st.session_state.manual_regions_a,
                            ref_size_a=st.session_state.ref_image_size_a,
                            manual_regions_b=st.session_state.manual_regions_b,
                            ref_size_b=st.session_state.ref_image_size_b,
                            # custom_bubbles 功能已移除
                        )
                        result["_key"] = key
                        result["_file_a"] = files["A"].name
                        result["_file_b"] = files["B"].name
                        
                        # 计分
                        std = st.session_state.standard_answers
                        if std:
                            correct = 0
                            detail = {}
                            for q, a in std.items():
                                sa = result["choices"].get(q, "")
                                sa_clean = sa.replace("(多涂)", "") if sa else ""
                                detail[q] = {
                                    "std": a,
                                    "ans": sa_clean,
                                    "raw": sa,
                                    "ok": sa_clean == a
                                }
                                if sa_clean == a:
                                    correct += 1
                            result["_score"] = correct
                            result["_total"] = len(std)
                            result["_detail"] = detail
                        else:
                            result["_score"] = 0
                            result["_total"] = 0
                            result["_detail"] = {}
                        
                        results.append(result)
                        bar.progress(int((idx+1)/len(valid)*100))
                    
                    status.empty()
                    bar.empty()
                    st.session_state.results = results
                    st.session_state.crop_results = []  # 清空手动裁剪结果
                    st.success(f"✅ 处理完成！共 {len(results)} 份答题卡")
                    
                    # 摘要
                    if results:
                        df = pd.DataFrame([{
                            "学生/文件": r["student_id"],
                            "条形码": r.get("barcode") or "未识别",
                            "选择题得分": f"{r.get('_score',0)}/{r.get('_total',0)}" if r.get('_total') else "未设答案",
                            "已识别": r["choice_count"],
                            "漏涂": r["empty_count"],
                            "多涂": r["multi_count"],
                        } for r in results])
                        st.dataframe(df)
    
    else:
        # ===== 手动区域裁剪模式 =====
        st.info("请先在「模板与参考」页面上传空白答题卡并分别标定A面和B面的截取区域。批量处理时会自动配对A+B图片并分别裁剪。")
        
        has_a = bool(st.session_state.manual_regions_a)
        has_b = bool(st.session_state.manual_regions_b)
        if not has_a and not has_b:
            st.error("⚠️ 请先标定截取区域（至少标定A面或B面）")
        else:
            col_status = st.columns(2)
            with col_status[0]:
                if has_a:
                    st.success(f"✅ A面已配置 {len(st.session_state.manual_regions_a)} 个区域")
                else:
                    st.warning("A面未配置区域")
            with col_status[1]:
                if has_b:
                    st.success(f"✅ B面已配置 {len(st.session_state.manual_regions_b)} 个区域")
                else:
                    st.warning("B面未配置区域")
            
            uploaded = st.file_uploader("上传需要裁剪的答题卡图片（A+B配对）", type=["jpg","jpeg","png"], accept_multiple_files=True, key="batch_crop")
            
            if uploaded:
                # 配对逻辑（与模板模式一致）
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
                
                valid = {k:v for k,v in pairs.items() if "A" in v and "B" in v}
                single_a = {k:v for k,v in pairs.items() if "A" in v and "B" not in v}
                single_b = {k:v for k,v in pairs.items() if "B" in v and "A" not in v}
                
                st.write(f"识别到 **{len(valid)}** 组A+B配对，**{len(single_a)}** 张单A面，**{len(single_b)}** 张单B面")
                
                if st.button("✂️ 开始批量裁剪", type="primary"):
                    output_dir = os.path.join("output", "manual_crop")
                    crop_results = []
                    
                    bar = st.progress(0)
                    status = st.empty()
                    total = len(valid) + len(single_a) + len(single_b)
                    processed = 0
                    
                    # 处理配对
                    for key, files in valid.items():
                        processed += 1
                        status.info(f"裁剪中 [{processed}/{total}]: {key} (A+B配对)")
                        
                        # A面
                        if has_a:
                            fa = files["A"]; fa.seek(0)
                            img_a = cv2.imdecode(np.asarray(bytearray(fa.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img_a is not None:
                                crops_a = CardProcessor.crop_by_regions(
                                    img_a, st.session_state.manual_regions_a, 
                                    output_dir, f"{key}_A",
                                    ref_size=st.session_state.ref_image_size_a
                                )
                                crop_results.append({
                                    "file": files["A"].name, "side": "A", "key": key, "crops": crops_a
                                })
                        
                        # B面
                        if has_b:
                            fb = files["B"]; fb.seek(0)
                            img_b = cv2.imdecode(np.asarray(bytearray(fb.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img_b is not None:
                                crops_b = CardProcessor.crop_by_regions(
                                    img_b, st.session_state.manual_regions_b, 
                                    output_dir, f"{key}_B",
                                    ref_size=st.session_state.ref_image_size_b
                                )
                                crop_results.append({
                                    "file": files["B"].name, "side": "B", "key": key, "crops": crops_b
                                })
                        
                        bar.progress(int(processed/total*100))
                    
                    # 处理单A面
                    for key, files in single_a.items():
                        processed += 1
                        status.info(f"裁剪中 [{processed}/{total}]: {key} (单A面)")
                        if has_a:
                            fa = files["A"]; fa.seek(0)
                            img_a = cv2.imdecode(np.asarray(bytearray(fa.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img_a is not None:
                                crops_a = CardProcessor.crop_by_regions(
                                    img_a, st.session_state.manual_regions_a, 
                                    output_dir, f"{key}_A",
                                    ref_size=st.session_state.ref_image_size_a
                                )
                                crop_results.append({
                                    "file": files["A"].name, "side": "A", "key": key, "crops": crops_a
                                })
                        bar.progress(int(processed/total*100))
                    
                    # 处理单B面
                    for key, files in single_b.items():
                        processed += 1
                        status.info(f"裁剪中 [{processed}/{total}]: {key} (单B面)")
                        if has_b:
                            fb = files["B"]; fb.seek(0)
                            img_b = cv2.imdecode(np.asarray(bytearray(fb.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                            if img_b is not None:
                                crops_b = CardProcessor.crop_by_regions(
                                    img_b, st.session_state.manual_regions_b, 
                                    output_dir, f"{key}_B",
                                    ref_size=st.session_state.ref_image_size_b
                                )
                                crop_results.append({
                                    "file": files["B"].name, "side": "B", "key": key, "crops": crops_b
                                })
                        bar.progress(int(processed/total*100))
                    
                    status.empty()
                    bar.empty()
                    st.session_state.crop_results = crop_results
                    st.session_state.results = []  # 清空模板识别结果
                    st.success(f"✅ 裁剪完成！共处理 {total} 份文件")
                    
                    # 摘要表格
                    if crop_results:
                        summary = []
                        for cr in crop_results:
                            for c in cr["crops"]:
                                summary.append({
                                    "原图": cr["file"],
                                    "面别": cr["side"],
                                    "区域名称": c["name"],
                                    "裁剪尺寸": f"{c['x2']-c['x1']}×{c['y2']-c['y1']}",
                                    "保存路径": c["path"]
                                })
                        st.dataframe(pd.DataFrame(summary))

# ---------- Tab 3: 结果核对与导出 ----------
with tab3:
    st.header("结果核对与导出")
    
    if st.session_state.process_mode == "模板识别模式":
        # ===== 原有流程 =====
        if not st.session_state.results:
            st.info("请先在「批量处理」页面上传并处理答题卡")
        else:
            results = st.session_state.results
            
            # 选择学生
            options = [f"{r['student_id']} ({r['_file_a']})" for r in results]
            sel_label = st.selectbox("选择答题卡查看详情", options)
            sel_idx = options.index(sel_label)
            result = results[sel_idx]
            
            st.subheader(f"📋 {result['student_id']} - 识别详情")
            
            c_left, c_right = st.columns([3,2])
            
            with c_left:
                st.markdown("**选择题识别结果（可人工修正）**")
                
                choice_data = []
                corrections = st.session_state.manual_corrections.get(result["student_id"], {})
                
                all_qs = sorted(result["choices"].keys())
                for q in all_qs:
                    auto_ans = result["choices"].get(q, "")
                    auto_display = auto_ans if auto_ans else "(未识别)"

                    std = st.session_state.standard_answers.get(q, "")
                    detail = result.get("_detail", {}).get(q, {})

                    # 是否已人工修正
                    corrected = corrections.get(q)
                    final_ans = corrected if corrected else (auto_ans.replace("(多涂)", "") if auto_ans else "")

                    status_str = ""
                    if std:
                        if final_ans == std:
                            status_str = "✅"
                        elif not final_ans:
                            status_str = "⚪"
                        else:
                            status_str = "❌"

                    choice_data.append({
                        "题号": q,
                        "自动识别": auto_display,
                        "人工修正": corrected if corrected else "",
                        "标准答案": std,
                        "状态": status_str,
                        "_q": q
                    })
                
                df_choice = pd.DataFrame(choice_data)
                
                # 使用data_editor让用户直接修改
                edited = st.data_editor(
                    df_choice[["题号", "自动识别", "人工修正", "标准答案", "状态"]],
                    column_config={
                        "人工修正": st.column_config.TextColumn("人工修正", help="如需修正，直接输入A/B/C/D等"),
                    },
                    hide_index=True,
                    height=500
                )
                
                # 保存修正
                new_corr = {}
                for _, row in edited.iterrows():
                    if row["人工修正"] and row["人工修正"].strip():
                        new_corr[int(row["题号"])] = row["人工修正"].strip().upper()
                st.session_state.manual_corrections[result["student_id"]] = new_corr
                
                # 计算最终得分
                if st.session_state.standard_answers:
                    final_score = 0
                    for q, std_ans in st.session_state.standard_answers.items():
                        corr = new_corr.get(q)
                        if corr:
                            if corr == std_ans:
                                final_score += 1
                        else:
                            auto = result["choices"].get(q, "")
                            if auto and auto.replace("(多涂)", "") == std_ans:
                                final_score += 1
                    st.metric("选择题最终得分", f"{final_score} / {len(st.session_state.standard_answers)}")
            
            with c_right:
                # 手动标定区域截图（所有类型都展示）
                manual_crops = result.get("manual_crops", [])
                if manual_crops:
                    st.markdown("**手动标定区域截图**")
                    type_order = ["选择题", "个人信息", "非选择题"]
                    type_labels = {"选择题": "📝 选择题区域", "个人信息": "🆔 个人信息区域", "非选择题": "✏️ 非选择题区域"}
                    for rtype in type_order:
                        crops_of_type = [c for c in manual_crops if c.get("type") == rtype]
                        if crops_of_type:
                            st.markdown(f"*{type_labels.get(rtype, rtype)}*")
                            cols = st.columns(min(2, len(crops_of_type)))
                            for idx, crop in enumerate(crops_of_type):
                                with cols[idx % len(cols)]:
                                    if Path(crop["path"]).exists():
                                        st.image(Image.open(crop["path"]),
                                                caption=f"{crop['region_name']} ({crop['side']}面)",
                                                use_column_width=True)
                                    else:
                                        st.warning(f"{crop['region_name']} 未找到")
                    st.divider()
                
                st.markdown("**主观题切分图片**")
                for subj in result.get("subjective", []):
                    qn = subj["q"]
                    path = subj["path"]
                    if Path(path).exists():
                        img = Image.open(path)
                        st.image(img, caption=f"{qn} (满分{subj['score']}分)", use_column_width=True)
                    else:
                        st.warning(f"{qn} 图片未找到")
                        with st.expander("调试信息"):
                            st.code(f"path: {path}\nexists: {Path(path).exists()}")
            
            # 全局导出
            st.divider()
            st.subheader("📥 批量导出")
            
            if st.button("生成Excel成绩单"):
                export_rows = []
                for r in results:
                    sid = r["student_id"]
                    corr = st.session_state.manual_corrections.get(sid, {})
                    
                    row = {
                        "学生ID": sid,
                        "条形码": r.get("barcode"),
                    }
                    
                    # 选择题成绩
                    if st.session_state.standard_answers:
                        score = 0
                        for q, sa in st.session_state.standard_answers.items():
                            ans = corr.get(q)
                            if not ans:
                                auto = r["choices"].get(q, "")
                                ans = auto.replace("(多涂)", "") if auto else ""
                            if ans == sa:
                                score += 1
                        row["选择题得分"] = score
                        row["选择题满分"] = len(st.session_state.standard_answers)
                    
                    # 每题答案（优先用人工修正）
                    for q in sorted(r["choices"].keys()):
                        ans = corr.get(q)
                        if not ans:
                            auto = r["choices"].get(q, "")
                            ans = auto if auto else ""
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
    
    else:
        # ===== 手动区域裁剪结果展示 =====
        if not st.session_state.crop_results:
            st.info("请先在「批量处理」页面选择「手动区域裁剪模式」并上传图片进行裁剪")
        else:
            crop_results = st.session_state.crop_results
            
            # 按 key 分组（A+B配对）
            from collections import defaultdict
            groups = defaultdict(list)
            for cr in crop_results:
                groups[cr.get("key", cr["file"])].append(cr)
            
            sel_key = st.selectbox("选择答题卡查看裁剪结果", sorted(groups.keys()))
            group_items = groups[sel_key]
            
            st.subheader(f"📁 {sel_key} - 裁剪结果")
            
            # 分别展示A面和B面
            for item in group_items:
                side = item["side"]
                st.markdown(f"**{side}面 - {item['file']}**")
                cols = st.columns(3)
                for idx, crop in enumerate(item["crops"]):
                    with cols[idx % 3]:
                        if Path(crop["path"]).exists():
                            img = Image.open(crop["path"])
                            st.image(img, caption=crop["name"], use_column_width=True)
                        else:
                            st.warning(f"{crop['name']} 未找到")
            
            # 全局汇总
            st.divider()
            st.subheader("📥 批量导出")
            
            all_summary = []
            for cr in crop_results:
                for c in cr["crops"]:
                    all_summary.append({
                        "原图": cr["file"],
                        "面别": cr["side"],
                        "区域名称": c["name"],
                        "裁剪路径": c["path"]
                    })
            
            if all_summary:
                df_all = pd.DataFrame(all_summary)
                st.dataframe(df_all)
                
                csv_buf = io.StringIO()
                df_all.to_csv(csv_buf, index=False, encoding="utf-8-sig")
                st.download_button(
                    label="⬇️ 下载裁剪汇总表(CSV)",
                    data=csv_buf.getvalue().encode("utf-8-sig"),
                    file_name="裁剪汇总.csv",
                    mime="text/csv"
                )
