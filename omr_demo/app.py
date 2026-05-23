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
        # 自定义选项框标定（绕过模板bubbles）
        "custom_bubbles": [],           # 用户自定义选项框列表
        "custom_bubbles_img_size": None,  # 标定时图片尺寸 (w, h)
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
    st.info("分别在上传的A面和B面空白答题卡上拖拽画出需要截取的区域，批量处理时会自动按A/B面对应裁剪。")
    
    # 选择标定基准图（必须是已上传的空白答题卡）
    available_refs = {}
    if st.session_state.blank_a is not None:
        available_refs["A面空白答题卡"] = st.session_state.blank_a
    if st.session_state.blank_b is not None:
        available_refs["B面空白答题卡"] = st.session_state.blank_b
    
    if not available_refs:
        st.warning("⚠️ 请先在上方的「上传空白答题卡」区域上传A面或B面空白答题卡")
    else:
        ref_choice = st.radio("选择标定基准图", list(available_refs.keys()), horizontal=True)
        is_a = "A面" in ref_choice
        regions_key = "manual_regions_a" if is_a else "manual_regions_b"
        regions_list = st.session_state[regions_key]
        key_prefix = "ra" if is_a else "rb"
        
        ref_img = available_refs[ref_choice]
        h, w = ref_img.shape[:2]
        size_key = "ref_image_size_a" if is_a else "ref_image_size_b"
        st.session_state[size_key] = (w, h)
        st.write(f"📐 图片尺寸：**{w} × {h}**")
        
        # 检测是否切换了基准图，如果是则自动清空画布
        prev_choice = st.session_state.get("canvas_ref_choice", "")
        if prev_choice != ref_choice:
            st.session_state.canvas_ref_choice = ref_choice
            st.session_state.canvas_version = st.session_state.get("canvas_version", 0) + 1
            st.rerun()
        
        # Canvas 显示尺寸（限制最大宽度 700，避免前端渲染问题）
        MAX_CANVAS_WIDTH = 700
        canvas_scale = MAX_CANVAS_WIDTH / w
        display_w = MAX_CANVAS_WIDTH
        display_h = int(h * canvas_scale)
        
        # 转为 PIL Image 传给 canvas
        bg_image = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
        bg_image = bg_image.convert("RGBA")
        bg_image = bg_image.resize((display_w, display_h))
        
        # 备用：直接用 st.image 显示图片（如果 Canvas 背景加载失败，用户可参考此图）
        st.image(bg_image, caption="参考图（若 Canvas 背景黑屏，请根据此图画框）", width=display_w)
        
        st.markdown(f"**🖱️ 在下方图片上拖拽画出截取区域（红色框），画完后点击「将画框添加为截取区域」：**")
        st.caption("提示：画得不满意可点击「清空画布」重新画；如要调整已添加区域的位置，可在下方列表中修改坐标。")
        
        canvas_version = st.session_state.get("canvas_version", 0)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=bg_image,
            height=display_h,
            width=display_w,
            drawing_mode="rect",
            key=f"canvas_{ref_choice}_{canvas_version}",
            update_streamlit=True,
        )
        
        st.divider()
        st.markdown("**✏️ 或手动输入区域坐标（若 Canvas 背景黑屏可用此方式）：**")
        st.caption("根据上方参考图，读取区域左上角和右下角的像素坐标")
        c_mx1, c_my1, c_mx2, c_my2 = st.columns(4)
        with c_mx1:
            mx1 = st.number_input("x1 (左)", min_value=0, value=0, key=f"manual_x1_{key_prefix}")
        with c_my1:
            my1 = st.number_input("y1 (上)", min_value=0, value=0, key=f"manual_y1_{key_prefix}")
        with c_mx2:
            mx2 = st.number_input("x2 (右)", min_value=0, value=100, key=f"manual_x2_{key_prefix}")
        with c_my2:
            my2 = st.number_input("y2 (下)", min_value=0, value=100, key=f"manual_y2_{key_prefix}")
        if st.button("➕ 添加手动坐标区域", key=f"manual_add_{key_prefix}"):
            regions_list.append({
                "name": f"区域{len(regions_list)+1}",
                "x1": int(mx1), "y1": int(my1), "x2": int(mx2), "y2": int(my2)
            })
            st.session_state[regions_key] = regions_list
            st.rerun()
        
        st.divider()
        
        # 提取 canvas 中的矩形并添加为区域
        if canvas_result.json_data is not None:
            rects = [obj for obj in canvas_result.json_data.get("objects", []) if obj.get("type") == "rect"]
            if rects:
                st.write(f"画布上检测到 **{len(rects)}** 个矩形框")
                col_add, col_clear = st.columns([1, 1])
                with col_add:
                    if st.button("➕ 将画框添加为截取区域"):
                        for obj in rects:
                            x1 = int(obj["left"] / canvas_scale)
                            y1 = int(obj["top"] / canvas_scale)
                            x2 = int((obj["left"] + obj["width"]) / canvas_scale)
                            y2 = int((obj["top"] + obj["height"]) / canvas_scale)
                            regions_list.append({
                                "name": f"区域{len(regions_list)+1}",
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2
                            })
                        st.session_state[regions_key] = regions_list
                        # 清空 canvas：通过更新 version 改变 key
                        st.session_state.canvas_version = canvas_version + 1
                        st.rerun()
                with col_clear:
                    if st.button("🗑️ 清空画布"):
                        st.session_state.canvas_version = canvas_version + 1
                        st.rerun()
            else:
                st.info("请在图片上拖拽画出矩形框")
        
        st.divider()
        st.markdown(f"**📌 当前{ref_choice}已添加的截取区域（可修改名称和坐标）**")
        
        def remove_region(idx):
            regions_list.pop(idx)
            st.session_state[regions_key] = regions_list
            for suffix in ["name", "x1", "y1", "x2", "y2"]:
                k = f"{key_prefix}_{idx}_{suffix}"
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
        
        # 读取当前区域配置
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
                    st.text_input("区域名称", value=region["name"], key=f"{key_prefix}_{i}_name")
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
                    st.button("❌ 删除", key=f"{key_prefix}_del_{i}", on_click=remove_region, args=(i,))
        else:
            st.info(f"暂无截取区域，请先在上方{ref_choice}图片中画框添加")
        
        # 实时同步回 session_state
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
        st.markdown(f"**🎨 {ref_choice}截取区域预览**")
        if synced:
            vis = ref_img.copy()
            h, w = vis.shape[:2]
            type_colors = {
                "非选择题": (0, 255, 0),    # 绿色
                "选择题": (255, 0, 0),      # 红色
                "个人信息": (0, 0, 255),    # 蓝色
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
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"{ref_choice}截取区域预览", use_column_width=True)
        else:
            st.info("请先添加截取区域")
    
    # 导出/导入配置
    if available_refs:
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
    
    st.divider()
    
    # ===== 标准答案识别 =====
    st.header("4. 标准答案识别")
    st.info("上传一张正确答案已填涂的答题卡A面图片，系统将自动识别标准答案并填充到左侧「标准答案设置」中。"
            "支持两种识别方式：①使用手动标定的「选择题」区域；②使用模板全局识别。")
    
    std_img_file = st.file_uploader("上传标准答案答题卡（A面，已填涂正确答案）", type=["jpg","jpeg","png"], key="std_answer_img")
    if std_img_file and st.session_state.processor:
        proc = st.session_state.processor
        bytes_std = np.asarray(bytearray(std_img_file.read()), dtype=np.uint8)
        std_img = cv2.imdecode(bytes_std, cv2.IMREAD_COLOR)
        
        # 标准答案识别时不使用差分法（避免空白答题卡与标准答案图片对齐偏差导致误判）
        saved_blank_refs = proc.blank_refs.copy()
        proc.blank_refs = {}
        
        # 允许用户调整识别阈值
        std_threshold = st.slider("识别阈值（标准答案）", 0.05, 0.50, 0.15, 0.01,
                                   help="选项框内黑色像素比例超过此阈值即认为已填涂。涂得较黑可用0.15，若识别不全可适当降低，若多涂太多可提高。")
        
        # 调试模式：显示每个选项的 fill_ratio
        show_debug = st.checkbox("显示调试信息（每个选项的识别置信度）", value=False)
        
        # 识别标准答案
        std_result = {}
        debug_data = {}
        
        # 优先使用手动标定的选择题区域
        manual_choice_regions = [r for r in st.session_state.manual_regions_a if r.get("type") == "选择题"]
        if manual_choice_regions:
            st.info(f"检测到 {len(manual_choice_regions)} 个手动标定的选择题区域，使用区域识别模式...")
            for region in manual_choice_regions:
                rc = proc.recognize_choices_in_region(std_img, region, "A", threshold=std_threshold, ref_size=st.session_state.ref_image_size_a, debug=show_debug)
                if show_debug and isinstance(rc, dict) and "answers" in rc:
                    std_result.update(rc["answers"])
                    debug_data.update(rc["debug"])
                else:
                    std_result.update(rc)
        else:
            st.info("未检测到手动标定的选择题区域，使用模板全局识别模式...")
            rc = proc.recognize_choices(std_img, "A", threshold=std_threshold, debug=show_debug)
            if show_debug and isinstance(rc, dict) and "answers" in rc:
                std_result = rc["answers"]
                debug_data = rc["debug"]
            else:
                std_result = rc
        
        # 恢复空白参考
        proc.blank_refs = saved_blank_refs
        
        # 过滤有效答案
        std_clean = {}
        multi_detected = []
        empty_detected = []
        for q, ans in std_result.items():
            if ans and not str(ans).endswith("(多涂)"):
                std_clean[q] = ans
            elif ans and str(ans).endswith("(多涂)"):
                multi_detected.append(f"Q{q}:{ans}")
            else:
                empty_detected.append(f"Q{q}")
        
        # 调试信息展示
        if show_debug and debug_data:
            st.markdown("**调试数据（每题各选项的 fill_ratio）**")
            debug_rows = []
            for q in sorted(debug_data.keys()):
                ratios = debug_data[q]["ratios"]
                filled = debug_data[q]["filled"]
                for opt in sorted(ratios.keys()):
                    debug_rows.append({
                        "题号": q,
                        "选项": opt,
                        "fill_ratio": f"{ratios[opt]:.3f}",
                        "是否超过阈值": "✅" if ratios[opt] > std_threshold else "",
                        "标准答案": "✓" if std_clean.get(q) == opt else ""
                    })
            st.dataframe(pd.DataFrame(debug_rows), height=400)
            st.caption(f"阈值={std_threshold}，fill_ratio > {std_threshold} 的选项会被判定为填涂")
        
        if std_clean:
            st.write(f"识别到 **{len(std_clean)}** 题标准答案，请核对：")
            
            metric_cols = st.columns(10)
            for idx, (q, ans) in enumerate(sorted(std_clean.items())):
                with metric_cols[idx % 10]:
                    st.metric(f"Q{q}", ans)
            
            if multi_detected:
                st.warning(f"以下题目检测到多涂（已排除）：{', '.join(multi_detected)}")
            if empty_detected:
                st.info(f"以下题目未识别到填涂（共{len(empty_detected)}题）：{', '.join(empty_detected)}")
            
            col_confirm, col_cancel = st.columns([1, 3])
            with col_confirm:
                if st.button("✅ 确认并设为标准答案", type="primary", key="confirm_std"):
                    st.session_state.standard_answers = std_clean
                    st.success(f"已设置 {len(std_clean)} 题标准答案！请到「批量处理」页面上传学生答题卡进行批改。")
                    st.rerun()
            
            # 可视化
            st.caption("下方为标准答案图片的识别可视化（绿色实心圆=识别为填涂的选项框中心，灰色空心圆=未填涂）：")
            vis_std = std_img.copy()
            h, w = vis_std.shape[:2]
            for b in proc.template["pages"]["A"].get("bubbles", []):
                bx, by = proc.scale_coords(b["x"], b["y"], w, h)
                q = b["q"]
                opt = b["opt"]
                if std_clean.get(q) == opt:
                    cv2.circle(vis_std, (bx, by), 8, (0, 255, 0), -1)
                else:
                    cv2.circle(vis_std, (bx, by), 4, (200, 200, 200), 1)
            st.image(cv2.cvtColor(vis_std, cv2.COLOR_BGR2RGB), caption="绿色实心圆=标准答案填涂位置", use_column_width=True)
        else:
            st.warning("未识别到有效标准答案，请检查：\n1. 图片是否清晰\n2. 正确答案是否已明显填涂\n3. 尝试调低「识别阈值」")


    st.divider()
    
    # ===== 自定义选项框标定（绕过模板bubbles）=====
    st.header("5. 自定义选项框标定（精准模式）")
    st.info("如果模板自动识别不准确，您可以在此上传标准答案图片，直接在图片上框选每个已填涂的选项框位置。批量处理时将使用您标定的位置进行识别。")
    
    custom_img_file = st.file_uploader("上传标准答案图片（用于标定选项框位置）", type=["jpg","jpeg","png"], key="custom_bubble_img")
    if custom_img_file:
        bytes_c = np.asarray(bytearray(custom_img_file.read()), dtype=np.uint8)
        custom_img = cv2.imdecode(bytes_c, cv2.IMREAD_COLOR)
        h, w = custom_img.shape[:2]
        st.session_state.custom_bubbles_img_size = (w, h)
        
        # Canvas 显示尺寸限制
        MAX_CANVAS_WIDTH = 700
        canvas_scale = MAX_CANVAS_WIDTH / w
        display_w = MAX_CANVAS_WIDTH
        display_h = int(h * canvas_scale)
        
        bg_image = Image.fromarray(cv2.cvtColor(custom_img, cv2.COLOR_BGR2RGB))
        bg_image = bg_image.convert("RGBA")
        bg_image = bg_image.resize((display_w, display_h))
        
        st.image(bg_image, caption="参考图（若 Canvas 背景黑屏，请根据此图画框）", width=display_w)
        
        st.markdown("**🖱️ 在下方图片上，在每个已填涂的选项框中心画小矩形框（红色）：**")
        st.caption("提示：框不需要很精确，框住选项即可。画完后在下方表格中填写对应的题号和选项。")
        
        canvas_ver = st.session_state.get("cb_canvas_version", 0)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=bg_image,
            height=display_h,
            width=display_w,
            drawing_mode="rect",
            key=f"custom_bubble_canvas_{canvas_ver}",
            update_streamlit=True,
        )
        
        if canvas_result.json_data is not None:
            rects = [obj for obj in canvas_result.json_data.get("objects", []) if obj.get("type") == "rect"]
            if rects:
                st.write(f"画布上检测到 **{len(rects)}** 个矩形框")
                
                # 转换为实际图片坐标
                bubble_list = []
                for idx, obj in enumerate(rects):
                    cx = int((obj["left"] + obj["width"] / 2) / canvas_scale)
                    cy = int((obj["top"] + obj["height"] / 2) / canvas_scale)
                    bubble_list.append({"idx": idx, "x": cx, "y": cy, "w": max(12, int(obj["width"] / canvas_scale)), "h": max(12, int(obj["height"] / canvas_scale))})
                
                st.markdown("**请为每个框填写题号和选项：**")
                st.caption("例如：框对应第1题的答案A，就填 题号=1，选项=A")
                
                updated_bubbles = []
                cols_per_row = 4
                for i in range(0, len(bubble_list), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx >= len(bubble_list):
                            break
                        b = bubble_list[idx]
                        with cols[j]:
                            st.write(f"框 {idx+1} (x={b['x']}, y={b['y']})")
                            q_val = st.number_input(f"题号", min_value=1, max_value=100, value=idx+1, key=f"cb_q_{idx}_{canvas_ver}")
                            opt_val = st.selectbox(f"选项", ["A","B","C","D","E","F","G"], key=f"cb_opt_{idx}_{canvas_ver}")
                            updated_bubbles.append({"q": int(q_val), "opt": opt_val, "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]})
                
                if st.button("💾 保存自定义选项框", type="primary"):
                    # 按题号去重：同一题保留用户最后设定的
                    dedup = {}
                    for b in updated_bubbles:
                        dedup[b["q"]] = b
                    st.session_state.custom_bubbles = list(dedup.values())
                    # 同时自动设为标准答案
                    std_from_custom = {b["q"]: b["opt"] for b in st.session_state.custom_bubbles}
                    st.session_state.standard_answers = std_from_custom
                    st.success(f"已保存 {len(st.session_state.custom_bubbles)} 个自定义选项框，并已设为标准答案！")
                    st.rerun()
                
                if st.button("🗑️ 清空画布重画"):
                    st.session_state.cb_canvas_version = canvas_ver + 1
                    st.rerun()
            else:
                st.info("请在图片上画出矩形框标定已填涂的选项位置")
    
    if st.session_state.custom_bubbles:
        st.write(f"当前已保存 **{len(st.session_state.custom_bubbles)}** 个自定义选项框：")
        df_cb = pd.DataFrame(st.session_state.custom_bubbles)
        st.dataframe(df_cb[["q", "opt", "x", "y"]])
        if st.button("🗑️ 清空自定义选项框"):
            st.session_state.custom_bubbles = []
            st.session_state.custom_bubbles_img_size = None
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
                            custom_bubbles=st.session_state.custom_bubbles,
                            custom_ref_size=st.session_state.custom_bubbles_img_size,
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
                
                for q in range(1, 46):
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
                    for q in range(1, 46):
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
