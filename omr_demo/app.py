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
    st.info("在A面和B面空白答题卡上直接拖拽画框，标定需要截取的区域。")

    has_a = st.session_state.blank_a is not None
    has_b = st.session_state.blank_b is not None

    if not has_a and not has_b:
        st.warning("请先在上方上传A面或B面空白答题卡")
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
                            st.selectbox("", ["非选择题", "选择题", "个人信息"],
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
    st.header("3. 黄金模板标定")
    st.info("上传一份**正确填涂**的答题卡，画出选择题列区域，系统自动识别答案并保存为黄金模板。后续批量处理时用此模板比对识别。")

    golden_img_file = st.file_uploader("上传正确填涂的答题卡", type=["jpg", "jpeg", "png"], key="golden_upload")

    if golden_img_file:
        bytes_data = np.asarray(bytearray(golden_img_file.read()), dtype=np.uint8)
        st.session_state.golden_image = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)

    if st.session_state.golden_image is not None:
        gimg = st.session_state.golden_image
        gh, gw = gimg.shape[:2]
        st.write(f"图片尺寸: {gw} × {gh}")

        # Canvas 画列框
        MAX_CANVAS_WIDTH = 700
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
                if g_canvas_result.json_data is not None:
                    rects = [obj for obj in g_canvas_result.json_data.get("objects", [])
                             if obj.get("type") == "rect"]
                    if rects:
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

    # 处理模式选择
    mode = st.radio(
        "选择处理模式",
        ["模板识别模式", "手动区域裁剪模式", "黄金模板对比模式"],
        index={"模板识别模式": 0, "手动区域裁剪模式": 1, "黄金模板对比模式": 2}.get(
            st.session_state.process_mode, 0),
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

    elif mode == "黄金模板对比模式":
        # ===== 黄金模板批量对比 =====
        if st.session_state.golden_template is None:
            st.warning("请先在「模板与参考」页面第三步「黄金模板标定」中生成黄金模板")
        else:
            st.info("上传待识别的答题卡（A面，需配对B面），系统自动用黄金模板比对识别")
            uploaded = st.file_uploader("批量上传", type=["jpg", "jpeg", "png"],
                                         accept_multiple_files=True, key="batch_golden")

            if uploaded:
                # 配对逻辑
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
                st.write(f"识别到 **{len(valid)}** 组有效答题卡（A+B配对）")

                debug_mode = st.checkbox("调试模式（输出每题详细采样值）", key="debug_golden")

                if st.button("开始黄金模板比对", type="primary"):
                    gtp = st.session_state.golden_template
                    results = []
                    bar = st.progress(0)
                    status = st.empty()

                    for idx, (key, files) in enumerate(valid.items()):
                        status.info(f"识别中 [{idx+1}/{len(valid)}]: {key}")

                        fa = files["A"]; fa.seek(0)
                        fb = files["B"]; fb.seek(0)
                        img_a = cv2.imdecode(np.asarray(bytearray(fa.read()), dtype=np.uint8), cv2.IMREAD_COLOR)
                        img_b = cv2.imdecode(np.asarray(bytearray(fb.read()), dtype=np.uint8), cv2.IMREAD_COLOR)

                        if img_a is None:
                            continue

                        result = gtp.recognize(img_a, debug=debug_mode)
                        result["_key"] = key
                        result["_file_a"] = files["A"].name
                        result["_file_b"] = files["B"].name

                        # 计分（与黄金答案对比）
                        correct = 0
                        total_ans = 0
                        for q, ans in result["answers"].items():
                            if ans.get("correct") is True:
                                correct += 1
                                total_ans += 1
                            elif ans.get("correct") is False:
                                total_ans += 1

                        result["_score"] = correct
                        result["_total"] = total_ans

                        # 主观题（复用原有裁剪）
                        if img_b is not None:
                            proc = st.session_state.processor
                            if proc:
                                result["subjective"] = proc.crop_subjective(
                                    img_b, key, "B", os.path.join("output", "subjective"))

                        results.append(result)
                        bar.progress(int((idx + 1) / len(valid) * 100))

                    status.empty()
                    bar.empty()
                    st.session_state.golden_results = results
                    st.session_state.results = results  # 复用 Tab3 展示
                    st.success(f"完成！共 {len(results)} 份")

                    if results:
                        df = pd.DataFrame([{
                            "学生/文件": r["_key"],
                            "得分": f"{r.get('_score', 0)}/{r.get('_total', 0)}",
                            "已作答": r["total"] - r["empty_count"],
                            "漏涂": r["empty_count"],
                            "多涂": r["multi_count"],
                            "异常": r.get("card_flag") or "-",
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
    
    if st.session_state.process_mode in ("模板识别模式", "黄金模板对比模式"):
        # ===== 模板识别 / 黄金模板对比 结果展示 =====
        if not st.session_state.results:
            st.info("请先在「批量处理」页面上传并处理答题卡")
        else:
            results = st.session_state.results

            is_golden = st.session_state.process_mode == "黄金模板对比模式"

            # 选择学生
            if is_golden:
                options = [f"{r['_key']} ({r['_file_a']})" for r in results]
            else:
                options = [f"{r['student_id']} ({r['_file_a']})" for r in results]
            sel_label = st.selectbox("选择答题卡查看详情", options)
            sel_idx = options.index(sel_label)
            result = results[sel_idx]
            
            if is_golden:
                student_label = result["_key"]
            else:
                student_label = result.get("student_id", result.get("_key", "unknown"))
            st.subheader(f"📋 {student_label} - 识别详情")

            # 显示调试信息
            debug_lines = result.get("debug_lines", [])
            if debug_lines:
                with st.expander(f"调试信息：{len(debug_lines)} 题未识别", expanded=True):
                    st.code("\n".join(debug_lines), language=None)

            c_left, c_right = st.columns([3, 2])

            with c_left:
                st.markdown("**选择题识别结果（可人工修正）**")

                choice_data = []
                sid = result.get("student_id", result.get("_key", "unknown"))
                corrections = st.session_state.manual_corrections.get(sid, {})

                if is_golden:
                    # 黄金模板格式: result["answers"] = {q: {answer, status, correct}}
                    all_qs = sorted(result["answers"].keys())
                else:
                    all_qs = sorted(result["choices"].keys())

                for q in all_qs:
                    if is_golden:
                        ans_info = result["answers"].get(q, {})
                        auto_ans = ans_info.get("answer", "")
                        auto_display = auto_ans if auto_ans else "(未识别)"
                        status_val = ans_info.get("status", "")
                        if status_val == "multi":
                            auto_display += "(多涂)"
                        std = st.session_state.golden_answers.get(q, "")
                    else:
                        auto_ans = result["choices"].get(q, "")
                        auto_display = auto_ans if auto_ans else "(未识别)"
                        std = st.session_state.standard_answers.get(q, "")

                    corrected = corrections.get(q)
                    final_ans = corrected if corrected else (auto_ans.replace("(多涂)", "") if auto_ans else "")

                    status_str = ""
                    if is_golden:
                        corr = ans_info.get("correct")
                        if corr is True:
                            status_str = "✅"
                        elif corr is False:
                            status_str = "❌"
                        elif auto_ans is None:
                            status_str = "⚪"
                    elif std:
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
                st.session_state.manual_corrections[sid] = new_corr

                # 计算最终得分
                if is_golden:
                    sc = result.get("_score", 0)
                    tot = result.get("_total", 0)
                    st.metric("选择题最终得分", f"{sc} / {tot}")
                elif st.session_state.standard_answers:
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
                    if is_golden:
                        sid = r.get("_key", "unknown")
                    else:
                        sid = r.get("student_id", r.get("_key", "unknown"))
                    corr = st.session_state.manual_corrections.get(sid, {})
                    
                    row = {
                        "学生ID": sid,
                        "条形码": r.get("barcode", ""),
                    }
                    
                    if is_golden:
                        row["选择题得分"] = r.get("_score", 0)
                        row["选择题满分"] = r.get("_total", 0)
                        row["异常标记"] = r.get("card_flag") or ""
                        answer_keys = sorted(r["answers"].keys())
                    else:
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
                        answer_keys = sorted(r["choices"].keys())
                    
                    # 每题答案
                    for q in answer_keys:
                        ans = corr.get(q)
                        if not ans:
                            if is_golden:
                                ans_info = r["answers"].get(q, {})
                                ans = ans_info.get("answer", "") or ""
                            else:
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
