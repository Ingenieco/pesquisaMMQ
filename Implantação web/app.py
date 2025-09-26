# --- Arreglo PROJ para Windows/conda (antes de importar geopandas/rasterio) ---
import os, sys
from pathlib import Path

def _set_proj_env():
    candidates = [
        Path(sys.prefix) / "Library" / "share" / "proj",  # conda (Win)
        Path(sys.prefix) / "share" / "proj",              # conda (Linux/Mac)
        Path(os.environ.get("CONDA_PREFIX", "")) / "Library" / "share" / "proj",
        Path(os.environ.get("CONDA_PREFIX", "")) / "share" / "proj",
    ]
    for p in candidates:
        try:
            if p and (p / "proj.db").exists():
                os.environ["PROJ_LIB"]  = str(p)
                os.environ["PROJ_DATA"] = str(p)
                # set_data_dir es opcional (no falla si pyproj no está aún)
                try:
                    from pyproj import datadir
                    datadir.set_data_dir(str(p))
                except Exception:
                    pass
                return str(p)
        except Exception:
            pass
    return None

_set_proj_env()
# --- FIN PROJ ---

import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import xy
from rasterio.io import MemoryFile
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx
import zipfile
import io
from PIL import Image

# =========================
# Lectura segura de la API Key
# =========================
def _get_rf_api_key() -> str:
    # 1) Streamlit Cloud / local con .streamlit/secrets.toml
    if "ROBOFLOW_API_KEY" in st.secrets:
        return st.secrets["ROBOFLOW_API_KEY"]
    # 2) Variable de entorno (local/despliegue)
    env = os.getenv("ROBOFLOW_API_KEY")
    if env:
        return env
    # 3) Fallar explícitamente si no existe
    raise RuntimeError(
        "Falta ROBOFLOW_API_KEY en st.secrets o en variable de entorno. "
        "Define la clave en Settings → Secrets (Streamlit Cloud) o exporta ROBOFLOW_API_KEY localmente."
    )

# ============== Config UI ==============
st.set_page_config(page_title="Detecção de Vendedores Ambulantes", layout="wide")
st.title("Detector de Vendedores Ambulantes em imágens aéreas de alta resolução")

# Banner/Logo en Sidebar (si existe logo.png junto al script)
logo_path = Path(__file__).parent / "logo.png"
if logo_path.exists():
    st.sidebar.image(str(logo_path), use_container_width=True)

# ===== Créditos (debajo de los filtros) =====
def sidebar_credits():
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Desenvolvido por **Carlos Gutiérrez** ©  \n"
        "[LinkedIn](https://www.linkedin.com/in/ingenieco-cegu/) · "
        "[GitHub](https://github.com/Ingenieco)"
    )

# ============== Modelo (cacheado) ==============
@st.cache_resource
def init_roboflow():
    rf = Roboflow(api_key=_get_rf_api_key())
    # Ajusta workspace/proyecto/versión aquí si cambian
    return rf.workspace("piscinas-2is0y").project("vendedores_ambulantes_pesquisa-pwao5").version(8).model

model = init_roboflow()

# ============== Utilidades ==============
def split_image(image_np, tile_size=256):
    h, w, _ = image_np.shape
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image_np[y:y+tile_size, x:x+tile_size]
            tiles.append(((x, y), tile))
    return tiles

def merge_predictions(predictions_tiles):
    merged = []
    for (x_off, y_off), preds in predictions_tiles:
        for p in preds.get('predictions', []):
            q = p.copy()
            q['x'] += x_off
            q['y'] += y_off
            merged.append(q)
    return merged

def draw_boxes(image_np, predictions, color=(45, 210, 90)):
    img = image_np.copy()
    for pred in predictions:
        x1 = int(pred['x'] - pred['width'] / 2)
        y1 = int(pred['y'] - pred['height'] / 2)
        x2 = int(x1 + pred['width'])
        y2 = int(y1 + pred['height'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"vendedor {pred['confidence']:.2f}"
        cv2.putText(img, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return img

def pixel_dets_to_gdf(predictions, transform, crs):
    if transform is None or crs is None:
        return None, None
    geoms, confs = [], []
    for p in predictions:
        row, col = float(p['y']), float(p['x'])  # y=row, x=col
        x_geo, y_geo = xy(transform, row, col)
        geoms.append(Point(x_geo, y_geo))
        confs.append(float(p['confidence']))
    gdf_src = gpd.GeoDataFrame({"confidence": confs}, geometry=geoms, crs=crs)
    gdf_wgs84 = gdf_src.to_crs(epsg=4326)
    return gdf_src, gdf_wgs84

def export_vector(gdf_wgs84, export_format):
    if export_format == "GeoJSON":
        data_bytes = gdf_wgs84.to_json().encode("utf-8")
        return data_bytes, "application/geo+json", "geojson"

    elif export_format == "Shapefile (.zip)":
        out_mem = io.BytesIO()
        out_dir = Path(st.session_state.get("tmp_dir", "/tmp")) / "shp_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        shp_path = out_dir / "deteccoes.shp"
        gdf_wgs84.to_file(shp_path, driver="ESRI Shapefile")
        with zipfile.ZipFile(out_mem, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(out_dir):
                zf.write(out_dir / fname, arcname=fname)
        return out_mem.getvalue(), "application/zip", "zip"

    else:  # GPKG por defecto
        out_dir = Path(st.session_state.get("tmp_dir", "/tmp"))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "deteccoes.gpkg"
        gdf_wgs84.to_file(out_path, driver="GPKG")
        with open(out_path, "rb") as f:
            data_bytes = f.read()
        return data_bytes, "application/geopackage+sqlite3", "gpkg"

def read_image_any(uploaded_file):
    content = uploaded_file.getvalue()

    # 1) Intentar con rasterio (georreferenciada)
    try:
        with MemoryFile(content) as memfile:
            with memfile.open() as src:
                transform = src.transform
                crs = src.crs
                if src.count >= 3:
                    arr = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
                else:
                    single = src.read(1)
                    arr = np.stack([single, single, single], axis=-1)

                # Escalado simple a 8 bits para mostrar e inferir
                arr = arr.astype(np.float32)
                mx = np.percentile(arr, 99.5)
                mx = mx if mx > 0 else arr.max() if arr.max() > 0 else 1.0
                arr = np.clip(arr / mx, 0, 1)
                rgb = (arr * 255).astype(np.uint8)
                return rgb, transform, crs
    except Exception:
        pass

    # 2) Fallback: imagen simple (no georreferenciada)
    bgr = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Não foi possível ler a imagem. Formato não suportado.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, None, None

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
    buf.seek(0)
    return buf.getvalue()

def image_to_png_bytes(img_rgb):
    pil = Image.fromarray(img_rgb)
    out = io.BytesIO()
    pil.save(out, format="PNG")
    out.seek(0)
    return out.getvalue()

# ============== Sidebar (filtros + créditos) ==============
confidence_threshold = st.sidebar.slider("Limiar de confiança (%)", 0, 100, 40)
tile_size = st.sidebar.selectbox("Tamanho do mosaico", [256, 384, 512], index=0)
export_format = st.sidebar.selectbox(
    "Formato de exportação GIS",
    ["GPKG", "GeoJSON", "Shapefile (.zip)"],  # GPKG por defecto
    index=0
)
sidebar_credits()

uploaded_file = st.file_uploader(
    "Escolha uma imagem (GeoTIFF ou outro formato). Se estiver georreferenciada, exportaremos pontos em WGS84.",
    type=['tif', 'tiff', 'png', 'jpg', 'jpeg']
)

# ============== Lógica principal con memoización por sesión ==============
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    uid = (uploaded_file.name, len(file_bytes), tile_size, int(confidence_threshold))

    if st.session_state.get("uid") != uid:
        st.session_state["uid"] = uid
        for k in ["img_rgb", "img_with_boxes", "png_bytes",
                  "predictions", "transform", "crs",
                  "gdf_wgs84", "gdf_src",
                  "vec_tuple", "csv_bytes",
                  "map_png_bytes"]:
            st.session_state.pop(k, None)

        image_np, transform, crs = read_image_any(uploaded_file)
        h, w, _ = image_np.shape

        with st.spinner("Analisando imagem com o modelo..."):
            conf = int(confidence_threshold)
            if h > tile_size or w > tile_size:
                tiles = split_image(image_np, tile_size)
                predictions_tiles = []
                for (x_off, y_off), tile in tiles:
                    preds = model.predict(tile, confidence=conf).json()
                    predictions_tiles.append(((x_off, y_off), preds))
                predictions = merge_predictions(predictions_tiles)
            else:
                preds = model.predict(image_np, confidence=conf).json()
                predictions = preds.get('predictions', [])

        img_with_boxes = draw_boxes(image_np, predictions)
        png_bytes = image_to_png_bytes(img_with_boxes)

        st.session_state["img_rgb"] = image_np
        st.session_state["img_with_boxes"] = img_with_boxes
        st.session_state["png_bytes"] = png_bytes
        st.session_state["predictions"] = predictions
        st.session_state["transform"] = transform
        st.session_state["crs"] = crs

        gdf_src, gdf_wgs84 = pixel_dets_to_gdf(predictions, transform, crs)
        st.session_state["gdf_src"] = gdf_src
        st.session_state["gdf_wgs84"] = gdf_wgs84

        if gdf_wgs84 is not None and len(gdf_wgs84) > 0:
            vec_bytes, vec_mime, vec_ext = export_vector(gdf_wgs84, export_format)
            st.session_state["vec_tuple"] = (vec_bytes, vec_mime, vec_ext)

            df_coords = gdf_wgs84.copy()
            df_coords["lon"] = df_coords.geometry.x
            df_coords["lat"] = df_coords.geometry.y
            csv_buf = io.StringIO()
            df_coords.drop(columns="geometry").to_csv(csv_buf, index=False)
            st.session_state["csv_bytes"] = csv_buf.getvalue().encode("utf-8")

            fig, ax = plt.subplots(figsize=(7, 7))
            try:
                gdf3857 = gdf_wgs84.to_crs(epsg=3857)
                gdf3857.plot(ax=ax, color="red", markersize=12, alpha=0.85)
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
                ax.set_axis_off()
                map_png = fig_to_png_bytes(fig)
                st.session_state["map_png_bytes"] = map_png
            except Exception:
                st.session_state["map_png_bytes"] = None
            plt.close(fig)
        else:
            st.session_state["vec_tuple"] = None
            st.session_state["csv_bytes"] = None
            st.session_state["map_png_bytes"] = None

    image_np = st.session_state["img_rgb"]
    img_with_boxes = st.session_state["img_with_boxes"]
    png_bytes = st.session_state["png_bytes"]
    predictions = st.session_state["predictions"]
    transform = st.session_state["transform"]
    crs = st.session_state["crs"]

    st.info(f"Tamanho: **{image_np.shape[1]}×{image_np.shape[0]}**  |  CRS: **{crs if crs else 'Não georreferenciada'}**")

    st.image(image_np, caption="Imagem carregada (RGB)", use_container_width=True)

    st.image(img_with_boxes, caption=f"Detecções: {len(predictions)}", use_container_width=True)
    st.download_button(
        "Descarregar imagem con deteccoes (PNG)",
        png_bytes,
        file_name="deteccoes.png",
        mime="image/png",
        type="primary",
        use_container_width=True
    )

    gdf_wgs84 = st.session_state["gdf_wgs84"]
    if gdf_wgs84 is not None and len(gdf_wgs84) > 0:
        st.subheader("Mapa de localizações (aproximadas)")
        map_png = st.session_state["map_png_bytes"]
        if map_png:
            st.image(map_png, caption="Pontos sobre o mapa base", use_container_width=True)
        else:
            st.warning("Não foi possível gerar o mapa base (sem conexão ou falha do provedor).")

        vec_tuple = st.session_state["vec_tuple"]
        if vec_tuple:
            vec_bytes, vec_mime, vec_ext = vec_tuple
            st.download_button(
                f"Descarregar detecções ({vec_ext.upper()})",
                vec_bytes,
                file_name=f"detecções.{vec_ext}",
                mime=vec_mime,
                use_container_width=True
            )

        st.subheader("Tabela de coordenadas (WGS84)")
        df_coords = gdf_wgs84.copy()
        df_coords["lon"] = df_coords.geometry.x
        df_coords["lat"] = df_coords.geometry.y
        st.dataframe(df_coords.drop(columns="geometry"), use_container_width=True)

        if st.session_state["csv_bytes"]:
            st.download_button(
                "Descarregar coordenadas (CSV)",
                st.session_state["csv_bytes"],
                file_name="deteccoes.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        if crs is None:
            st.warning("A imagem não possui georreferência. A imagem foi gerada com caixas, mas não há exportação em formato GIS.")
        else:
            st.info("Não houve detecções com o limite configurado.")
